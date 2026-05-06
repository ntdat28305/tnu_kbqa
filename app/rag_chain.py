"""
app/rag_chain.py
Hybrid RAG chain — 3 nguồn retrieval:
  1. KG search      — NetworkX graph (quan hệ văn bản, không cần Neo4j)
  2. Vector search  — ChromaDB + multilingual-e5-base
  3. BM25 search    — từ data/processed/ JSON

Groq rotation + Gemini fallback.
"""
from __future__ import annotations

import gc
import itertools
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from flashrank import Ranker, RerankRequest

load_dotenv()

PROCESSED_DIR  = Path("data/processed")
RAW_DIR        = Path("data/raw")
KG_DIR         = Path("data/kg")
EMBED_MODEL    = "intfloat/multilingual-e5-base"
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION     = "tnu_aiqa"


# ── Groq rotation ─────────────────────────────────────────────

GROQ_API_KEYS = [k for k in [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
] if k]
if not GROQ_API_KEYS:
    key = os.getenv("GROQ_API_KEY")
    if key:
        GROQ_API_KEYS = [key]

GROQ_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct", 
    "llama-3.3-70b-versatile",                   
    "qwen/qwen3-32b",                            
    "llama-3.1-8b-instant",                     
]

# Sắp xếp combinations theo thứ tự ưu tiên model (xịn → nhỏ)
# Mỗi câu hỏi mới sẽ bắt đầu từ model đầu tiên (70b)
# Ưu tiên model xịn trước, xài hết 3 key mới xuống model thấp hơn
_groq_combinations = [
    (api_key, model)
    for model in GROQ_MODELS
    for api_key in GROQ_API_KEYS
]
_current_combo_idx = 0

print(f"🔑 Groq keys: {len(GROQ_API_KEYS)} | Models: {len(GROQ_MODELS)} | Combos: {len(_groq_combinations)}")


def get_groq_llm_at(idx: int):
    """Lấy LLM tại vị trí idx trong combinations."""
    if not _groq_combinations:
        return None, "none"
    api_key, model = _groq_combinations[idx % len(_groq_combinations)]
    return ChatGroq(api_key=api_key, model=model,
                    temperature=0.1, max_tokens=2048,
                    max_retries=0), f"groq/{model}"


def get_current_groq_llm():
    return get_groq_llm_at(_current_combo_idx)


def rotate_groq():
    global _current_combo_idx
    _current_combo_idx = (_current_combo_idx + 1) % len(_groq_combinations)
    _, model = _groq_combinations[_current_combo_idx]
    print(f"🔄 Groq rotate → {model}")


# Gemini đã bị tắt — dùng Groq rotation thay thế


# ── Lazy singletons ───────────────────────────────────────────

_embeddings:     HuggingFaceEmbeddings | None = None
_vectorstore:    Chroma | None                = None
_bm25_retriever: BM25Retriever | None         = None
_ranker:         Ranker | None                = None
_kg_graph                                     = None
_kg_doc_index:   dict | None                  = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        print(f"📥 Loading embedding model: {EMBED_MODEL}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vectorstore() -> QdrantVectorStore | None:
    global _vectorstore
    if _vectorstore is None:
        if not QDRANT_URL or not QDRANT_API_KEY:
            print("⚠️  Thiếu QDRANT_URL hoặc QDRANT_API_KEY trong .env")
            return None
        try:
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            count  = client.count(COLLECTION).count
            _vectorstore = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION,
                embedding=get_embeddings(),
            )
            print(f"✅ Qdrant loaded: {count} vectors")
        except Exception as e:
            print(f"⚠️  Qdrant error: {e}")
            return None
    return _vectorstore


def get_bm25_retriever() -> BM25Retriever | None:
    global _bm25_retriever
    if _bm25_retriever is not None:
        return _bm25_retriever

    # Đọc từ raw/ (processed/ không có plain_text)
    search_dirs = [RAW_DIR, PROCESSED_DIR]
    docs = []

    for search_dir in search_dirs:
        files = sorted(search_dir.rglob("*.json"))
        if files:
            for f in files:
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    text = data.get("plain_text", "")
                    meta = data.get("metadata", {})
                    if text and len(text) > 50:
                        docs.append(Document(
                            page_content=text,
                            metadata={
                                "title":      meta.get("title", ""),
                                "doc_type":   meta.get("doc_type", ""),
                                "doc_number": meta.get("doc_number", ""),
                                "issued_by":  meta.get("issued_by", ""),
                                "article_id": meta.get("article_id", f.stem),
                            },
                        ))
                except Exception:
                    pass
            break  # dùng dir đầu tiên tìm thấy

    if not docs:
        print("⚠️  Không có dữ liệu cho BM25")
        return None

    _bm25_retriever = BM25Retriever.from_documents(docs, k=4)
    print(f"✅ BM25 built: {len(docs)} docs")
    return _bm25_retriever


def get_kg():
    """Load NetworkX KG (lazy). Trả về (G, doc_index)."""
    global _kg_graph, _kg_doc_index
    if _kg_graph is not None:
        return _kg_graph, _kg_doc_index

    kg_file    = KG_DIR / "knowledge_graph.json"
    index_file = KG_DIR / "doc_index.json"

    if not kg_file.exists():
        print("⚠️  KG chưa build — chạy: python -m pipeline.kg_build.kg_builder")
        _kg_graph, _kg_doc_index = None, {}
        return None, {}

    try:
        import networkx as nx
        _kg_graph = nx.node_link_graph(
            json.loads(kg_file.read_text(encoding="utf-8")),
            directed=True,
        )
        _kg_doc_index = {}
        if index_file.exists():
            _kg_doc_index = json.loads(index_file.read_text(encoding="utf-8"))
        print(f"✅ KG loaded: {_kg_graph.number_of_nodes()} nodes, "
              f"{_kg_graph.number_of_edges()} edges")
    except Exception as e:
        print(f"⚠️  KG load error: {e}")
        _kg_graph, _kg_doc_index = None, {}

    return _kg_graph, _kg_doc_index


def get_ranker() -> Ranker:
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    return _ranker


# ── KG search (NetworkX) ──────────────────────────────────────

def _normalize_vn(text: str) -> str:
    """Chuẩn hóa tiếng Việt có dấu → không dấu để so sánh."""
    import unicodedata
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.replace("đ", "d").replace(" ", "_")


def kg_search(query: str, top_k: int = 4) -> list[Document]:
    """
    1. Tìm nodes trong KG khớp keyword (có dấu + không dấu)
    2. Lấy các văn bản liên quan qua graph traversal
    3. Load plain_text từ processed/raw JSON
    """
    G, doc_index = get_kg()
    if G is None:
        return []

    docs  = []
    seen  = set()
    words     = [w for w in query.split() if len(w) > 2][:6]
    words_norm = [_normalize_vn(w) for w in words]

    # Tìm nodes khớp keyword — so sánh cả có dấu lẫn không dấu
    matched_ids = set()
    for aid, attrs in G.nodes(data=True):
        title      = (attrs.get("title") or "").lower()
        docnum     = (attrs.get("doc_number") or "").lower()
        aid_norm   = _normalize_vn(aid)
        title_norm = _normalize_vn(title)

        for w, wn in zip(words, words_norm):
            if (w in title or w in docnum or
                wn in aid_norm or wn in title_norm):
                matched_ids.add(aid)
                break

    # Mở rộng qua neighbors (depth=1)
    expand = set()
    for aid in matched_ids:
        expand.update(G.successors(aid))
        expand.update(G.predecessors(aid))
    all_ids = matched_ids | expand

    # Load text
    for aid in list(all_ids)[:top_k * 2]:
        if aid in seen or aid.startswith("ext::"):
            continue
        seen.add(aid)
        text = _load_text(aid)
        if not text:
            continue
        meta = doc_index.get(aid, {})
        # Thêm context quan hệ
        rel_ctx = _get_relation_context(G, doc_index, aid)
        docs.append(Document(
            page_content=text[:3000],
            metadata={
                "title":       meta.get("title", aid),
                "doc_type":    meta.get("doc_type", ""),
                "doc_number":  meta.get("doc_number", ""),
                "issued_by":   meta.get("issued_by", ""),
                "article_id":  aid,
                "source":      "kg",
                "kg_relations": rel_ctx,
            },
        ))
        if len(docs) >= top_k:
            break

    # Thêm relation context docs
    rel_docs = _build_relation_docs(G, doc_index, query)
    print(f"📌 KG: {len(docs)} docs + {len(rel_docs)} relation docs")
    return docs + rel_docs


def _get_relation_context(G, doc_index: dict, aid: str) -> str:
    """Tạo chuỗi mô tả quan hệ của văn bản với các văn bản khác."""
    lines = []
    for nbr in list(G.successors(aid))[:5]:
        if nbr.startswith("ext::"):
            continue
        edge = G.get_edge_data(aid, nbr) or {}
        nbr_title = (doc_index.get(nbr) or {}).get("title", nbr)
        lines.append(f"→ [{edge.get('rel_type','?')}] {nbr_title[:60]}")
    for nbr in list(G.predecessors(aid))[:5]:
        if nbr.startswith("ext::"):
            continue
        edge = G.get_edge_data(nbr, aid) or {}
        nbr_title = (doc_index.get(nbr) or {}).get("title", nbr)
        lines.append(f"← [{edge.get('rel_type','?')}] {nbr_title[:60]}")
    return "\n".join(lines)


def _build_relation_docs(G, doc_index: dict, query: str) -> list[Document]:
    """Tạo Document chứa thông tin quan hệ giữa các văn bản liên quan đến query."""
    words = [w for w in query.split() if len(w) > 3][:3]
    rel_texts = []

    for aid, attrs in G.nodes(data=True):
        title = (attrs.get("title") or "").lower()
        for w in words:
            if w.lower() in title:
                for nbr in G.successors(aid):
                    if nbr.startswith("ext::"):
                        continue
                    edge       = G.get_edge_data(aid, nbr) or {}
                    src_title  = attrs.get("title", aid)
                    nbr_attrs  = G.nodes[nbr]
                    nbr_title  = nbr_attrs.get("title", nbr)
                    rel_texts.append(
                        f"{src_title[:50]} --[{edge.get('rel_type','?')}]--> {nbr_title[:50]}"
                    )
                break

    if not rel_texts:
        return []

    return [Document(
        page_content="Quan hệ giữa các văn bản liên quan:\n" + "\n".join(rel_texts[:15]),
        metadata={"source": "kg_relations"},
    )]


def _load_text(article_id: str) -> str:
    """Load plain_text từ processed/ hoặc raw/."""
    for search_dir in [PROCESSED_DIR, RAW_DIR]:
        for f in search_dir.rglob(f"{article_id}.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                return data.get("plain_text", "")
            except Exception:
                pass
    return ""


# ── Vector search ─────────────────────────────────────────────

def vector_search(query: str, top_k: int = 4) -> list[Document]:
    vs = get_vectorstore()
    if vs is None:
        return []
    try:
        results = vs.similarity_search(f"query: {query}", k=top_k)
        for doc in results:
            doc.page_content = doc.page_content.replace("passage: ", "", 1)
        print(f"🔍 Vector: {len(results)} docs")
        return results
    except Exception as e:
        print(f"⚠️  Vector error: {e}")
        return []


# ── BM25 search ───────────────────────────────────────────────

def bm25_search(query: str) -> list[Document]:
    retriever = get_bm25_retriever()
    if retriever is None:
        return []
    try:
        docs = retriever.invoke(query)
        print(f"📄 BM25: {len(docs)} docs")
        return docs
    except Exception as e:
        print(f"⚠️  BM25 error: {e}")
        return []


# ── Rerank ────────────────────────────────────────────────────

def rerank(query: str, docs: list[Document], top_k: int = 6) -> list[Document]:
    if not docs:
        return []
    try:
        passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
        request  = RerankRequest(query=query, passages=passages)
        results  = get_ranker().rerank(request)
        return [docs[r["id"]] for r in results[:top_k]]
    except Exception as e:
        print(f"⚠️  Rerank error: {e}")
        return docs[:top_k]


# ── Prompts ───────────────────────────────────────────────────

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Bạn là trợ lý AI về kiểm định chất lượng giáo dục đại học Việt Nam (TNU-AIQA).

NGUYÊN TẮC:
- CHỈ trả lời dựa trên TÀI LIỆU THAM KHẢO được cung cấp
- Khi trích dẫn, ghi rõ tên văn bản và số điều/khoản nếu có (ví dụ: "Theo Điều 5, Thông tư 20/2026")
- Nếu có thông tin quan hệ văn bản (phần "Quan hệ giữa các văn bản") → tích hợp vào câu trả lời
- Nếu không có trong tài liệu → "Tôi không tìm thấy thông tin này trong tài liệu TNU-AIQA."
- Trả lời tiếng Việt, tự nhiên, dễ đọc
- Dùng **bold** cho tiêu đề quan trọng, danh sách có số thứ tự khi liệt kê
- KHÔNG trả ra raw text dạng pipe | một cách thô"""),
    ("user", "TÀI LIỆU THAM KHẢO:\n{context}\n\nCÂU HỎI: {question}\n\nTrả lời:")
])

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Bạn là trợ lý AI của TNU-AIQA.
Tóm tắt nội dung theo cấu trúc:
1. **Chủ đề chính**: 1-2 câu
2. **Những điểm nổi bật**: 3-5 điểm
3. **Kết luận**: 1-2 câu
Chỉ dùng thông tin từ tài liệu."""),
    ("user", "NỘI DUNG:\n{context}\n\nTóm tắt:"),
])

SUMMARY_KEYWORDS = [
    "tóm tắt", "tổng hợp", "nội dung bài", "bài viết về",
    "summarize", "tóm lược", "overview", "giới thiệu bài",
]


def detect_summary_intent(question: str) -> bool:
    return any(kw in question.lower() for kw in SUMMARY_KEYWORDS)


# ── Hybrid search ─────────────────────────────────────────────

def hybrid_search(query: str, top_k: int = 6) -> tuple[str, list[str]]:
    """
    KG (NetworkX) + Vector + BM25 → Dedup → Rerank → Context string
    """
    kg_docs     = kg_search(query, top_k=4)
    vector_docs = vector_search(query, top_k=4)
    bm25_docs   = bm25_search(query)

    all_docs = kg_docs + vector_docs + bm25_docs

    # Dedup theo 80 ký tự đầu
    seen, unique = set(), []
    for d in all_docs:
        key = d.page_content[:80]
        if key not in seen:
            seen.add(key)
            unique.append(d)

    # Rerank
    if len(unique) > top_k:
        unique = rerank(query, unique, top_k=top_k * 2)

    # Luôn giữ kg_relation docs (ngắn, cung cấp context quan hệ)
    final, final_keys = [], set()
    for d in unique:
        if d.metadata.get("source") == "kg_relations":
            key = d.page_content[:80]
            if key not in final_keys:
                final_keys.add(key)
                final.insert(0, d)  # ưu tiên đầu

    for d in unique:
        key = d.page_content[:80]
        if key not in final_keys and len(final) < top_k:
            final_keys.add(key)
            final.append(d)

    if not final:
        return "", []

    # Thêm kg_relations từ metadata vào context
    context_parts = []
    for d in final:
        part = d.page_content
        kg_rel = d.metadata.get("kg_relations", "")
        if kg_rel:
            part += f"\n\n[Quan hệ với văn bản khác]\n{kg_rel}"
        context_parts.append(part)

    context = "\n---\n".join(context_parts)[:12000]
    sources = list(dict.fromkeys(
        d.metadata.get("title", "")[:100]
        for d in final
        if d.metadata.get("title") and d.metadata.get("source") != "kg_relations"
    ))

    print(f"✅ Final: {len(final)} chunks, {len(context)} chars, {len(sources)} sources")
    return context, sources


# ── Main RAG ──────────────────────────────────────────────────

def run_rag(question: str) -> tuple[str, str]:
    import time
    context, _ = hybrid_search(question)

    if not context.strip():
        return "Tôi không tìm thấy thông tin này trong tài liệu TNU-AIQA.", "none"

    prompt  = SUMMARY_PROMPT if detect_summary_intent(question) else PROMPT
    n       = len(_groq_combinations)
    attempt = 0

    while True:
        idx = attempt % n
        try:
            llm, model_name = get_groq_llm_at(idx)
            chain  = prompt | llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": question})
            gc.collect()
            return answer, model_name
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ["429", "rate", "quota",
                                       "401", "invalid", "authentication",
                                       "decommissioned", "deprecated"]):
                attempt += 1
                print(f"🔄 [{attempt}] rotate → {_groq_combinations[attempt % n][1]}")
                # Sau 1 vòng đầy → chờ 10s rồi thử lại từ đầu
                if attempt % n == 0:
                    print(f"⏳ Hết {n} combos, chờ 10s...")
                    time.sleep(10)
                continue
            raise e


def run_rag_stream(question: str):
    """
    Streaming version của run_rag.
    Yield từng token ngay khi LLM generate.
    """
    context, sources = hybrid_search(question)

    if not context.strip():
        yield {"type": "token", "content": "Tôi không tìm thấy thông tin này trong tài liệu TNU-AIQA."}
        yield {"type": "done", "sources": [], "model_used": "none"}
        return

    prompt = SUMMARY_PROMPT if detect_summary_intent(question) else PROMPT

    import time
    n       = len(_groq_combinations)
    attempt = 0

    while True:
        idx = attempt % n
        try:
            llm, model_name = get_groq_llm_at(idx)
            chain = prompt | llm
            for chunk in chain.stream({"context": context, "question": question}):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    yield {"type": "token", "content": token}
            yield {"type": "done", "sources": sources, "model_used": model_name}
            return
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ["429", "rate", "quota",
                                       "401", "invalid", "authentication",
                                       "decommissioned", "deprecated"]):
                attempt += 1
                print(f"🔄 stream [{attempt}] rotate → {_groq_combinations[attempt % n][1]}")
                if attempt % n == 0:
                    print(f"⏳ Hết {n} combos, chờ 10s...")
                    time.sleep(10)
                continue
            yield {"type": "error", "content": str(e)}
            return
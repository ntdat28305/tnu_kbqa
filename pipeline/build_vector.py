"""
pipeline/build_vector.py
Build vector store trên Qdrant Cloud thay ChromaDB local.

Chạy từ root project:
    python -m pipeline.build_vector
    python -m pipeline.build_vector --reset   # xóa collection cũ trước
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

RAW_DIR        = Path("data/raw")
EMBED_MODEL    = "intfloat/multilingual-e5-base"
COLLECTION     = "tnu_aiqa"
BATCH_SIZE     = 50

QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def get_embeddings() -> HuggingFaceEmbeddings:
    print(f"📥 Loading embedding model: {EMBED_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_docs_from_raw() -> list[Document]:
    """Load tất cả JSON từ data/raw/ → LangChain Documents."""
    docs  = []
    files = sorted(RAW_DIR.rglob("*.json"))

    if not files:
        print(f"⚠️  Không có file trong {RAW_DIR}/")
        return docs

    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            text = data.get("plain_text", "").strip()
            meta = data.get("metadata", {})

            if len(text) < 80:
                continue

            docs.append(Document(
                page_content=f"passage: {text}",
                metadata={
                    "title":      meta.get("title", "")[:200],
                    "doc_type":   meta.get("doc_type", ""),
                    "doc_number": meta.get("doc_number", ""),
                    "issued_by":  meta.get("issued_by", ""),
                    "article_id": meta.get("article_id", f.stem),
                    "source":     meta.get("source", ""),
                    "filename":   meta.get("filename", ""),
                },
            ))
        except Exception as e:
            print(f"⚠️  Lỗi {f.name}: {e}")

    print(f"✅ Loaded {len(docs)} docs từ {RAW_DIR}/")
    return docs


def chunk_docs(docs: list[Document]) -> list[Document]:
    """Chunk văn bản dài — giữ nguyên văn bản ngắn."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter_law = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200,
        separators=["\n## ", "\n[BẢNG]", "\n\n", "\n", ".", " "],
    )
    splitter_gen = RecursiveCharacterTextSplitter(
        chunk_size=3000, chunk_overlap=300,
        separators=["\n\n", "\n", ".", " "],
    )
    LAW_TYPES = {"luat","thong_tu","nghi_quyet","quyet_dinh","nghi_dinh","ket_luan","quy_dinh"}

    chunks = []
    for doc in docs:
        dt = doc.metadata.get("doc_type", "")
        sp = splitter_law if dt in LAW_TYPES else splitter_gen
        chunks.extend(sp.split_documents([doc]))

    filtered = [c for c in chunks if len(c.page_content.strip()) >= 80]
    print(f"✅ Chunks: {len(filtered)}")
    return filtered


def build(reset: bool = False):
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Thiếu QDRANT_URL hoặc QDRANT_API_KEY trong .env")

    print(f"🔗 Kết nối Qdrant Cloud: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Reset collection nếu cần
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION in collections:
        if reset:
            client.delete_collection(COLLECTION)
            print(f"🗑️  Đã xóa collection '{COLLECTION}'")
        else:
            count = client.count(COLLECTION).count
            print(f"ℹ️  Collection '{COLLECTION}' đã tồn tại ({count} vectors)")
            ans = input("Xóa và build lại? (y/N): ").strip().lower()
            if ans == "y":
                client.delete_collection(COLLECTION)
                print(f"🗑️  Đã xóa collection '{COLLECTION}'")
            else:
                print("Bỏ qua — dùng collection hiện tại")
                return

    # Load + chunk docs
    docs   = load_docs_from_raw()
    chunks = chunk_docs(docs)

    if not chunks:
        print("⚠️  Không có chunks để index")
        return

    # Load embedding model
    embeddings = get_embeddings()

    # Tạo collection với đúng vector size (multilingual-e5-base = 768)
    print(f"📐 Tạo collection '{COLLECTION}' (dim=768)")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    # Upload theo batch
    print(f"📤 Upload {len(chunks)} chunks → Qdrant Cloud (batch={BATCH_SIZE})")
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
    )

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        vectorstore.add_documents(batch)
        print(f"  ✅ {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")

    total = client.count(COLLECTION).count
    print(f"\n✅ Xong! Collection '{COLLECTION}': {total} vectors trên Qdrant Cloud")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Xóa collection cũ trước khi build")
    args = parser.parse_args()
    build(reset=args.reset)
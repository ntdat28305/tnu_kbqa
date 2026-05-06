"""
pipeline/kg_build/kg_builder.py
Build lightweight Knowledge Graph bằng NetworkX — KHÔNG cần Neo4j.

Graph lưu quan hệ giữa các văn bản pháp luật dựa trên:
  1. Quan hệ "Căn cứ/dẫn chiếu" trích từ phần đầu văn bản
  2. Quan hệ entities dùng chung (entities trùng tên ở nhiều văn bản)

Lưu ra: data/kg/knowledge_graph.json + data/kg/doc_index.json

Chạy từ root project:
    python -m pipeline.kg_build.kg_builder --dryrun
    python -m pipeline.kg_build.kg_builder
    python -m pipeline.kg_build.kg_builder --reset
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import networkx as nx

from utils.logger import get_pipeline_logger

logger     = get_pipeline_logger("kg_builder")
PROCESSED  = Path("data/processed")
RAW        = Path("data/raw")
KG_DIR     = Path("data/kg")
KG_FILE    = KG_DIR / "knowledge_graph.json"
DOC_INDEX  = KG_DIR / "doc_index.json"


# ── Trích quan hệ "Căn cứ" từ plain_text ─────────────────────

REF_PATTERNS = [
    # "Căn cứ Thông tư số 20/2026/TT-BGDĐT"
    r"[Cc]ăn cứ\s+([^\n;,\.]{10,120})",
    # "Theo Thông tư 04/2025"
    r"[Tt]heo\s+((?:Thông tư|Nghị định|Luật|Quyết định|Nghị quyết)[^\n;,\.]{5,80})",
    # "thay thế ... Quyết định số ..."
    r"thay thế\s+([^\n;,\.]{10,100})",
]

DOC_NUM_PATTERN = re.compile(
    r"\b(\d{1,4}/\d{4}/(?:QH|NĐ|TT|QĐ|NQ|CT|BGDĐT|BGDDT|CP|TTg)[A-Z0-9/\-]*)",
    re.IGNORECASE,
)


def extract_references(text: str) -> list[str]:
    """Trích các văn bản được dẫn chiếu từ 1500 ký tự đầu."""
    head  = text[:1500]
    found = set()

    for pattern in REF_PATTERNS:
        for m in re.finditer(pattern, head):
            snippet = m.group(1).strip()
            # Lấy số văn bản nếu có
            nums = DOC_NUM_PATTERN.findall(snippet)
            for n in nums:
                found.add(n.upper())
            # Lấy tên văn bản rút gọn nếu không có số
            if not nums and len(snippet) > 8:
                found.add(snippet[:80])

    return list(found)


def normalize_doc_num(s: str) -> str:
    return re.sub(r"\s+", "", s).upper()


# ── Build Graph ───────────────────────────────────────────────

def build_graph(files: list[Path]) -> tuple[nx.DiGraph, dict]:
    """
    Trả về:
      G         — DiGraph với nodes = article_id, edges = quan hệ
      doc_index — dict {article_id: metadata}
    """
    G         = nx.DiGraph()
    doc_index = {}
    # Mapping doc_number → article_id để lookup cross-reference
    docnum_to_id: dict[str, str] = {}

    # Pass 1: nạp tất cả nodes + đánh index doc_number
    for f in files:
        try:
            data     = json.loads(f.read_text(encoding="utf-8"))
            meta     = data.get("metadata", {})
            aid      = meta.get("article_id", f.stem)
            title    = meta.get("title", "")
            doc_type = meta.get("doc_type", "")
            doc_num  = meta.get("doc_number", "")
            issued   = meta.get("issued_by", "")
            # plain_text nằm trong raw/, không phải processed/
            text     = data.get("plain_text", "")
            if not text:
                raw_candidates = list(RAW.rglob(f"{aid}.json"))
                if raw_candidates:
                    try:
                        raw_data = json.loads(raw_candidates[0].read_text(encoding="utf-8"))
                        text = raw_data.get("plain_text", "")
                    except Exception:
                        pass

            G.add_node(aid, title=title, doc_type=doc_type,
                       doc_number=doc_num, issued_by=issued,
                       text_len=len(text))

            doc_index[aid] = {
                "article_id": aid, "title": title,
                "doc_type": doc_type, "doc_number": doc_num,
                "issued_by": issued,
            }

            if doc_num:
                docnum_to_id[normalize_doc_num(doc_num)] = aid

        except Exception as e:
            logger.warning(f"Pass1 lỗi {f.name}: {e}")

    logger.info(f"Graph nodes: {G.number_of_nodes()}")

    # Pass 2: thêm edges từ "Căn cứ" references
    ref_edges = 0
    for f in files:
        try:
            data  = json.loads(f.read_text(encoding="utf-8"))
            meta  = data.get("metadata", {})
            src   = meta.get("article_id", f.stem)
            text  = data.get("plain_text", "")
            if not text:
                raw_candidates = list(RAW.rglob(f"{src}.json"))
                if raw_candidates:
                    try:
                        raw_data = json.loads(raw_candidates[0].read_text(encoding="utf-8"))
                        text = raw_data.get("plain_text", "")
                    except Exception:
                        pass
            refs  = extract_references(text)

            for ref in refs:
                norm = normalize_doc_num(ref)
                tgt  = docnum_to_id.get(norm)
                if tgt and tgt != src:
                    G.add_edge(src, tgt, rel_type="CAN_CU", ref_text=ref[:80])
                    ref_edges += 1
                else:
                    # Văn bản ngoài corpus — thêm node placeholder
                    placeholder = f"ext::{norm[:40]}"
                    if not G.has_node(placeholder):
                        G.add_node(placeholder, title=ref[:80],
                                   doc_type="external", external=True)
                    if not G.has_edge(src, placeholder):
                        G.add_edge(src, placeholder,
                                   rel_type="CAN_CU_NGOAI", ref_text=ref[:80])
                        ref_edges += 1
        except Exception as e:
            logger.warning(f"Pass2 lỗi {f.name}: {e}")

    logger.info(f"Graph edges (căn cứ): {ref_edges}")

    # Pass 3: edges từ entities dùng chung
    entity_to_docs: dict[str, list[str]] = {}
    for f in files:
        try:
            data  = json.loads(f.read_text(encoding="utf-8"))
            meta  = data.get("metadata", {})
            aid   = meta.get("article_id", f.stem)
            nodes = data.get("nodes", [])
            for nd in nodes:
                name = nd.get("name", "").strip()
                if len(name) > 3:
                    entity_to_docs.setdefault(name, [])
                    if aid not in entity_to_docs[name]:
                        entity_to_docs[name].append(aid)
        except Exception:
            pass

    shared_edges = 0
    for entity, aids in entity_to_docs.items():
        if len(aids) < 2:
            continue
        for i in range(len(aids)):
            for j in range(i + 1, len(aids)):
                a, b = aids[i], aids[j]
                if not G.has_edge(a, b):
                    G.add_edge(a, b, rel_type="SHARE_ENTITY", entity=entity)
                    G.add_edge(b, a, rel_type="SHARE_ENTITY", entity=entity)
                    shared_edges += 2

    logger.info(f"Graph edges (share_entity): {shared_edges}")
    logger.info(f"Total edges: {G.number_of_edges()}")
    return G, doc_index


# ── Serialize / Deserialize ───────────────────────────────────

def save_graph(G: nx.DiGraph, doc_index: dict):
    KG_DIR.mkdir(parents=True, exist_ok=True)

    data = nx.node_link_data(G)
    KG_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    DOC_INDEX.write_text(json.dumps(doc_index, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    logger.info(f"Saved KG → {KG_FILE}")
    logger.info(f"Saved doc index → {DOC_INDEX}")


def load_graph() -> tuple[nx.DiGraph, dict]:
    """Load KG từ file JSON. Trả về (G, doc_index)."""
    if not KG_FILE.exists():
        return nx.DiGraph(), {}
    G = nx.node_link_graph(
        json.loads(KG_FILE.read_text(encoding="utf-8")),
        directed=True,
    )
    doc_index = {}
    if DOC_INDEX.exists():
        doc_index = json.loads(DOC_INDEX.read_text(encoding="utf-8"))
    return G, doc_index


# ── Query helpers (dùng trong rag_chain.py) ───────────────────

def get_related_docs(G: nx.DiGraph, doc_index: dict,
                     article_id: str, depth: int = 2) -> list[dict]:
    """
    Trả về danh sách các văn bản liên quan đến article_id
    trong vòng `depth` bước trên graph.
    """
    if article_id not in G:
        return []
    related = []
    visited = {article_id}
    queue   = [(article_id, 0)]

    while queue:
        curr, d = queue.pop(0)
        if d >= depth:
            continue
        for nbr in list(G.successors(curr)) + list(G.predecessors(curr)):
            if nbr in visited or nbr.startswith("ext::"):
                continue
            visited.add(nbr)
            edge_data = G.get_edge_data(curr, nbr) or G.get_edge_data(nbr, curr) or {}
            info = doc_index.get(nbr, {"article_id": nbr})
            info["_rel_type"] = edge_data.get("rel_type", "RELATED")
            info["_depth"]    = d + 1
            related.append(info)
            queue.append((nbr, d + 1))

    return related


def find_docs_by_keyword(G: nx.DiGraph, doc_index: dict,
                         keyword: str) -> list[dict]:
    """
    Tìm các article_id trong graph có title/doc_number chứa keyword.
    """
    kw = keyword.lower()
    results = []
    for aid, attrs in G.nodes(data=True):
        title  = (attrs.get("title") or "").lower()
        docnum = (attrs.get("doc_number") or "").lower()
        if kw in title or kw in docnum:
            results.append(doc_index.get(aid, {"article_id": aid, **attrs}))
    return results


# ── Pipeline ──────────────────────────────────────────────────

def run(limit: int | None = None, dry_run: bool = False, reset: bool = False):
    logger.info("=" * 50)
    logger.info("KG Builder — NetworkX (no Neo4j)")
    logger.info("=" * 50)

    if reset and KG_DIR.exists():
        import shutil
        shutil.rmtree(KG_DIR)
        logger.warning(f"Đã xoá {KG_DIR}/")

    files = sorted(PROCESSED.rglob("*.json"))
    if not files:
        logger.error(f"Không có file trong {PROCESSED}/ — hãy chạy entity_extractor trước")
        return

    if limit:
        files = files[:limit]
    logger.info(f"Xử lý {len(files)} files")

    G, doc_index = build_graph(files)

    if dry_run:
        logger.info("[DRY RUN] Không lưu file")
        logger.info(f"  Nodes: {G.number_of_nodes()}")
        logger.info(f"  Edges: {G.number_of_edges()}")
        for u, v, d in list(G.edges(data=True))[:10]:
            logger.info(f"  {u} --[{d.get('rel_type')}]--> {v}")
        return

    save_graph(G, doc_index)
    logger.info("=== Xong ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",  type=int,  default=None)
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--reset",  action="store_true",
                        help="Xoá KG cũ trước khi build lại")
    args = parser.parse_args()
    run(limit=args.limit, dry_run=args.dryrun, reset=args.reset)
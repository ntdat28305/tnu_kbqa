"""
pipeline/kg_build/entity_extractor.py
Extract entities + relations từ data/raw/ dùng Groq LLM.
Schema FULL DYNAMIC — LLM tự quyết định node label và relation type.

Khi rate limit: tự động rotate sang model khác (cùng 1 API key).

Chạy từ root project:
    python -m pipeline.kg_build.entity_extractor --dryrun
    python -m pipeline.kg_build.entity_extractor --limit 5
    python -m pipeline.kg_build.entity_extractor
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from utils.logger import get_pipeline_logger

load_dotenv()
logger = get_pipeline_logger("entity_extractor")

RAW_DIR    = Path("data/raw")
CHECKPOINT = Path("data/checkpoint.txt")
OUTPUT_DIR = Path("data/processed")

# ── Danh sách model rotate khi rate limit ─────────────────────
# Thứ tự ưu tiên: mạnh nhất → nhanh nhất
GROQ_MODELS = [
    "llama-3.3-70b-versatile",           # 12K TPM, 100K/day — dùng trước
    "meta-llama/llama-4-scout-17b-16e-instruct",  # 30K TPM, 500K/day — fallback 1
    "llama-3.1-8b-instant",              # 6K TPM, 500K/day  — fallback 2
    "openai/gpt-oss-120b",               # 8K TPM, 200K/day  — fallback 3
    "qwen/qwen3-32b",                    # 6K TPM, 500K/day  — fallback 4
]


# ── Groq model rotator ────────────────────────────────────────

class GroqRotator:
    """
    Dùng 1 API key, rotate qua nhiều model khi bị rate limit.
    Sau khi thử hết model → chờ 60s rồi quay lại model đầu.
    """

    def __init__(self, api_key: str):
        self.api_key     = api_key
        self.models      = GROQ_MODELS
        self.model_index = 0
        self.client      = Groq(api_key=api_key)
        logger.info(f"Dùng model: {self.current_model}")

    @property
    def current_model(self) -> str:
        return self.models[self.model_index]

    def next_model(self) -> bool:
        """
        Chuyển sang model kế tiếp.
        Trả về True nếu còn model khác, False nếu đã hết vòng.
        """
        next_index = (self.model_index + 1) % len(self.models)
        rotated    = next_index != 0          # False = đã đi hết 1 vòng
        self.model_index = next_index
        logger.warning(f"Rate limit — chuyển sang model: {self.current_model}")
        return rotated

    def chat(self, system: str, user: str, retries: int = 5) -> str:
        """
        Gửi request với retry + model rotation.
        Logic:
          - Lỗi rate_limit / 429 / tokens → next_model(), sleep ngắn
          - Hết 1 vòng model            → sleep 60s (chờ quota reset)
          - Lỗi khác                    → sleep ngắn, thử lại cùng model
        """
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.current_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    temperature=0,
                    max_tokens=2048,
                )
                return resp.choices[0].message.content or ""

            except Exception as e:
                err = str(e).lower()

                if "rate_limit" in err or "429" in err or \
                   "tokens" in err or "quota" in err:
                    still_have_models = self.next_model()
                    if not still_have_models:
                        # Đã thử hết tất cả model → chờ quota reset
                        wait = 60
                        logger.warning(
                            f"Đã thử hết {len(self.models)} models — "
                            f"chờ {wait}s để quota reset..."
                        )
                        time.sleep(wait)
                    else:
                        time.sleep(3)

                elif any(k in err for k in ["model", "not found",
                                              "decommissioned", "deprecated",
                                              "invalid_request"]):
                    # Model không tồn tại / bị khai tử → sang model kế
                    logger.warning(f"Model không hợp lệ/khai tử: {self.current_model}")
                    self.next_model()
                    time.sleep(1)

                else:
                    logger.error(f"Groq lỗi attempt {attempt + 1}: {e}")
                    time.sleep(2)

        logger.error("Đã thử hết retries — bỏ qua chunk này")
        return ""


# ── Khởi tạo Groq ─────────────────────────────────────────────

def get_groq() -> GroqRotator:
    api_key = os.getenv("GROQ_API_KEY_1") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Không tìm thấy GROQ_API_KEY_1 trong .env")
    logger.info("Loaded API key")
    return GroqRotator(api_key)


# ── Prompt ────────────────────────────────────────────────────

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích văn bản pháp luật giáo dục đại học Việt Nam.
Nhiệm vụ: trích xuất knowledge graph từ đoạn văn bản.

QUY TẮC NODE:
- "label": loại thực thể. Ví dụ: VanBan, ToChuc, CoSoGiaoDuc, ChinhSach,
  MucTieu, LinhVuc, GiangVien, NguoiHoc, VungLanhTho, ChiTieu ...
  Bạn ĐƯỢC TỰ ĐẶT label mới nếu không có label nào phù hợp.
- "name": tên cụ thể của thực thể trong văn bản.
- "props": dict thuộc tính bổ sung (có thể {}).

QUY TẮC RELATION:
- "type": tên quan hệ UPPER_SNAKE_CASE. Ví dụ: BAN_HANH, QUY_DINH,
  DAT_MUCTIEU, GIAO_NHIEM_VU, THUOC_VUNG, AP_DUNG_CHO, LIEN_QUAN ...
  Bạn ĐƯỢC TỰ ĐẶT relation type mới nếu cần.
- "from" và "to": phải là "name" của node đã khai báo.
- "props": dict thuộc tính bổ sung (có thể {}).

CHỈ trả về JSON hợp lệ, không giải thích thêm."""

USER_TEMPLATE = """Văn bản:
\"\"\"
{text}
\"\"\"

Trả về JSON:
{{
  "nodes": [
    {{"label": "LoaiThucThe", "name": "Tên cụ thể", "props": {{}}}}
  ],
  "relations": [
    {{"from": "Tên A", "type": "LOAI_QUAN_HE", "to": "Tên B", "props": {{}}}}
  ]
}}"""


# ── Chunking ──────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            cut = text.rfind("\n\n", start, end)
            if cut == -1:
                cut = text.rfind(".", start, end)
            if cut != -1 and cut > start + chunk_size // 2:
                end = cut + 1
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# ── Parse JSON ────────────────────────────────────────────────

def parse_json(raw: str) -> dict:
    if not raw:
        return {}
    # Bỏ markdown code fence
    raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = raw.replace("```", "").strip()

    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return {}

    candidate = raw[start:end]

    # Thử parse trực tiếp
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Repair: cắt tại dấu } hợp lệ cuối cùng
    # (model nhỏ hay bị truncate JSON giữa chừng)
    depth  = 0
    last_valid_end = -1
    in_str = False
    escape = False
    for i, ch in enumerate(candidate):
        if escape:
            escape = False
            continue
        if ch == chr(92) and in_str:   # backslash
            escape = True
            continue
        if ch == '"' and not escape:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_valid_end = i + 1
                break

    if last_valid_end > 0:
        try:
            result = json.loads(candidate[:last_valid_end])
            logger.debug("JSON repaired thành công")
            return result
        except json.JSONDecodeError:
            pass

    logger.warning("JSON parse lỗi — bỏ qua chunk này")
    return {}


# ── Merge nhiều chunk ─────────────────────────────────────────

def merge_results(results: list[dict]) -> dict:
    node_map: dict[tuple, dict] = {}
    rel_set:  set[tuple]        = set()
    rel_list: list[dict]        = []

    for r in results:
        for node in r.get("nodes", []):
            if not isinstance(node, dict):
                continue
            label = str(node.get("label") or "Entity").strip()
            name  = str(node.get("name")  or "").strip()
            if not name:
                continue
            key = (label.lower(), name.lower())
            if key not in node_map:
                node_map[key] = {"label": label, "name": name,
                                 "props": node.get("props") or {}}
            else:
                node_map[key]["props"].update(node.get("props") or {})

        for rel in r.get("relations", []):
            if not isinstance(rel, dict):
                continue
            frm   = str(rel.get("from") or "").strip()
            rtype = str(rel.get("type") or "RELATED_TO").strip().upper()
            to    = str(rel.get("to")   or "").strip()
            if not frm or not to:
                continue
            key = (frm.lower(), rtype, to.lower())
            if key not in rel_set:
                rel_set.add(key)
                rel_list.append({"from": frm, "type": rtype,
                                 "to": to, "props": rel.get("props", {})})

    return {"nodes": list(node_map.values()), "relations": rel_list}


# ── Extract một file ──────────────────────────────────────────

# Chunk size theo từng model — model nhỏ cần chunk nhỏ hơn
MODEL_CHUNK_SIZE = {
    "llama-3.3-70b-versatile":                     2000,
    "meta-llama/llama-4-scout-17b-16e-instruct":   2000,
    "llama-3.1-8b-instant":                        1000,  # nhỏ hơn
    "openai/gpt-oss-120b":                         2000,
    "qwen/qwen3-32b":                              1500,
}


def extract_file(groq: GroqRotator, title: str, text: str) -> dict:
    chunk_size = MODEL_CHUNK_SIZE.get(groq.current_model, 1500)
    chunks  = chunk_text(f"Tiêu đề: {title}\n\n{text}",
                         chunk_size=chunk_size)
    results = []
    for j, chunk in enumerate(chunks):
        logger.debug(f"  chunk {j+1}/{len(chunks)} ({len(chunk)} chars) [{groq.current_model}]")
        raw = groq.chat(SYSTEM_PROMPT, USER_TEMPLATE.format(text=chunk))
        r   = parse_json(raw)
        if r:
            results.append(r)
        time.sleep(1.0)
    return merge_results(results) if results else {}


# ── Checkpoint ────────────────────────────────────────────────

def load_checkpoint() -> set[str]:
    if not CHECKPOINT.exists():
        return set()
    done = set(CHECKPOINT.read_text(encoding="utf-8").splitlines())
    logger.info(f"Checkpoint: {len(done)} files đã xử lý")
    return done


def save_checkpoint(file_id: str):
    with open(CHECKPOINT, "a", encoding="utf-8") as f:
        f.write(file_id + "\n")


# ── Save ──────────────────────────────────────────────────────

def save_processed(filepath: Path, metadata: dict, result: dict):
    category = metadata.get("category", metadata.get("doc_type", "misc"))
    out_dir  = OUTPUT_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filepath.with_suffix(".json").name
    out_path.write_text(
        json.dumps({"metadata": metadata, **result},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── Pipeline ──────────────────────────────────────────────────

def run(limit: int | None = None, dry_run: bool = False):
    logger.info("=" * 50)
    logger.info("Entity Extractor — GDĐH domain")
    logger.info("=" * 50)

    if not RAW_DIR.exists():
        logger.error(f"Không tìm thấy {RAW_DIR}/ — hãy chạy loader trước")
        return

    groq = get_groq()
    done = load_checkpoint()

    all_files = sorted(RAW_DIR.rglob("*.json"))
    pending   = [f for f in all_files if f.stem not in done]
    if limit:
        pending = pending[:limit]

    total = len(pending)
    logger.info(f"Cần xử lý: {total} files (đã xong: {len(done)})")

    success, errors = 0, 0

    for i, filepath in enumerate(pending, 1):
        logger.info(f"[{i}/{total}] {filepath.parent.name}/{filepath.name}")
        try:
            data     = json.loads(filepath.read_text(encoding="utf-8"))
            metadata = data.get("metadata", {})
            title    = metadata.get("title", filepath.stem)
            text     = data.get("plain_text", "")
        except Exception as e:
            logger.error(f"Lỗi đọc: {e}")
            errors += 1
            continue

        if not text:
            logger.warning("Không có text — bỏ qua")
            errors += 1
            continue

        result = extract_file(groq, title, text)
        if not result or not result.get("nodes"):
            logger.warning("Không extract được")
            errors += 1
            continue

        nodes     = result["nodes"]
        rels      = result["relations"]
        labels    = {n["label"] for n in nodes}
        rel_types = {r["type"]  for r in rels}

        if dry_run:
            logger.info(f"  [DRY RUN] nodes={len(nodes)} labels={labels}")
            logger.info(f"            rels={len(rels)} types={rel_types}")
        else:
            save_processed(filepath, metadata, result)
            save_checkpoint(filepath.stem)
            success += 1
            logger.info(
                f"  OK — {len(nodes)} nodes ({len(labels)} labels) | "
                f"{len(rels)} rels ({len(rel_types)} types) | "
                f"model: {groq.current_model}"
            )

    logger.info(f"=== Xong: {success} success, {errors} errors ===")


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",  type=int, default=None,
                        help="Giới hạn số file xử lý")
    parser.add_argument("--dryrun", action="store_true",
                        help="Chỉ in log, không lưu")
    args = parser.parse_args()
    run(limit=args.limit, dry_run=args.dryrun)
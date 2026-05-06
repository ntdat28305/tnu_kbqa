"""
pipeline/ingestion/loader.py
Load dữ liệu từ 2 nguồn:
  1. Blog Blogger (tnu-aiqa.blogspot.com) — qua JSON feed
  2. DOCX — dùng unstructured để giữ cấu trúc bảng

Chạy từ root project:
    python -m pipeline.ingestion.loader
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

BLOG_URL = "https://tnu-aiqa.blogspot.com"

CONTENT_ELEMENTS = {
    "Title", "Header", "NarrativeText", "Text",
    "ListItem", "Table", "FigureCaption",
}

FORM_FILENAME_KEYWORDS = [
    "mau_don", "mau don", "bieu_mau", "bieu mau",
    "don_de_nghi", "don de nghi", "phieu_dang_ky",
    "bang_diem", "mau_phieu", "phu_luc_mau",
]

FORM_CONTENT_SIGNALS = [
    "họ và tên sinh viên",
    "mã sinh viên",
    "người làm đơn",
    "ký và ghi rõ họ tên",
    "đắk lắk, ngày......tháng",
]

FORM_KEYWORDS = FORM_FILENAME_KEYWORDS


# ── Helpers ───────────────────────────────────────────────────

def _detect_doc_type(text: str, filename: str = "") -> str:
    fname = filename.lower()
    head  = text[:200].lower()

    if any(k in fname for k in ["thong_tu", "thong tu", "tt-bgddt", "tt_bgddt"]):
        return "thong_tu"
    if any(k in fname for k in ["nghi_dinh", "nghi dinh", "nd-cp"]):
        return "nghi_dinh"
    if any(k in fname for k in ["luat", "luật"]):
        return "luat"
    if any(k in fname for k in ["quy_dinh", "quy dinh"]):
        return "quy_dinh"
    if any(k in fname for k in ["nghi_quyet", "nq-"]):
        return "nghi_quyet"

    can_cu_pos = head.find("căn cứ")
    if can_cu_pos > 0:
        head = head[:can_cu_pos]

    if any(k in head for k in ["thông tư", "thong tu"]):     return "thong_tu"
    if any(k in head for k in ["nghị quyết", "nghi quyet"]): return "nghi_quyet"
    if any(k in head for k in ["quyết định", "quyet dinh"]): return "quyet_dinh"
    if any(k in head for k in ["nghị định", "nghi dinh"]):   return "nghi_dinh"
    if any(k in head for k in ["quy định", "quy dinh"]):     return "quy_dinh"
    if "luật" in head or "luat" in head:                      return "luat"
    if any(k in head for k in ["kết luận", "ket luan"]):      return "ket_luan"
    return "tai_lieu"


def _extract_doc_number(text: str) -> str:
    patterns = [
        r"\b\d{1,4}/\d{4}/(?:QH|NĐ|TT|QĐ|NQ|CT|BGDĐT|BGDDT|CP|TTg)[A-Z0-9/\-]*",
        r"\b\d{1,4}[-–](?:NQ|CT|KL|TB)/TW\b",
    ]
    for p in patterns:
        m = re.search(p, text[:500], re.IGNORECASE)
        if m:
            return m.group(0)
    return ""


def _extract_issued_by(text: str) -> str:
    known = [
        "Quốc hội", "Bộ Chính trị", "Ban Chấp hành Trung ương",
        "Thủ tướng Chính phủ", "Chính phủ",
        "Bộ Giáo dục và Đào tạo", "Bộ Khoa học và Công nghệ",
        "Ủy ban nhân dân", "Trường Đại học Tây Nguyên",
    ]
    for org in known:
        if org.lower() in text[:500].lower():
            return org
    return ""


def _is_form(filename: str, content: str) -> bool:
    fname_lower   = filename.lower()
    content_lower = content.lower()

    if any(kw in fname_lower for kw in FORM_FILENAME_KEYWORDS):
        return True

    has_form_signal = any(sig in content_lower for sig in FORM_CONTENT_SIGNALS)
    if not has_form_signal:
        return False

    has_legal_structure = any(marker in content_lower for marker in [
        "điều 1.", "điều 2.", "chương i", "chương ii",
        "khoản 1.", "khoản 2.",
    ])
    return has_form_signal and not has_legal_structure


def _clean_content(content: str) -> str:
    """
    Làm sạch content:
    1. Bảng header trang bìa (Quốc hội | Cộng hòa...) → text thường
    2. Gộp nhiều dòng trống
    """
    HEADER_KEYWORDS = [
        "quốc hội", "cộng hòa xã hội", "độc lập", "tự do", "hạnh phúc",
        "luật số", "nghị định số", "thông tư số", "quyết định số",
        "hà nội, ngày", "đắk lắk, ngày",
    ]

    lines = content.split("\n")
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "[BẢNG]" and len("\n".join(result)) < 500:
            table_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != "[/BẢNG]":
                table_lines.append(lines[i])
                i += 1
            table_text = "\n".join(table_lines).lower()
            is_header = any(kw in table_text for kw in HEADER_KEYWORDS)
            if is_header:
                for tl in table_lines:
                    for cell in tl.split(" | "):
                        cell = cell.strip()
                        if cell and len(cell) > 2:
                            result.append(cell)
            else:
                result.append("[BẢNG]")
                result.extend(table_lines)
                result.append("[/BẢNG]")
        else:
            result.append(line)
        i += 1

    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(result))
    return cleaned.strip()


def _inject_headings(content: str) -> str:
    """
    Với DOCX dùng toàn style Normal (văn bản pháp lý VN),
    inject ## heading trước Chương/Điều/Mục để chunker tách đúng.
    """
    lines = content.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^Chương\s+([IVXivx\d]+)\s*$", stripped):
            result.append(f"\n## {stripped}")
        elif re.match(r"^CHƯƠNG\s+([IVXivx\d]+)\s*$", stripped):
            result.append(f"\n## {stripped}")
        elif re.match(r"^Điều\s+\d+[\.\s]", stripped):
            result.append(f"\n## {stripped}")
        elif re.match(r"^Mục\s+\d+[\.\s]", stripped):
            result.append(f"\n## {stripped}")
        else:
            result.append(line)
    return "\n".join(result)


# ── Blog loader ───────────────────────────────────────────────

def _parse_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "iframe"]):
        tag.decompose()
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            table.replace_with("\n" + "\n".join(rows) + "\n")
    for li in soup.find_all("li"):
        li.insert_before("• ")
    return soup.get_text(separator="\n").strip()


def load_blog(max_posts: int = 200) -> list[Document]:
    docs, seen = [], set()
    feed_url   = f"{BLOG_URL}/feeds/posts/default?alt=json&max-results={max_posts}"
    try:
        entries = requests.get(feed_url, timeout=15).json().get("feed", {}).get("entry", [])
        for entry in entries:
            title = entry.get("title", {}).get("$t", "").strip()
            if title in seen:
                continue
            seen.add(title)

            content = _parse_html(entry.get("content", {}).get("$t", ""))
            if len(content) < 100:
                print(f"⚠️  Skip blog (ngắn): {title[:50]}")
                continue

            url = next(
                (l["href"] for l in entry.get("link", []) if l.get("rel") == "alternate"),
                BLOG_URL,
            )
            doc_type = _detect_doc_type(title + " " + content[:200], "")
            docs.append(Document(
                page_content=content,
                metadata={
                    "title":      title,      "source":     "blog",
                    "doc_type":   doc_type,   "url":        url,
                    "doc_number": _extract_doc_number(title + "\n" + content),
                    "issued_by":  _extract_issued_by(content),
                    "is_form":    False,
                },
            ))
            print(f"✅ Blog: {title[:60]} ({len(content)} chars)")
            time.sleep(0.3)

        print(f"✅ Tổng blog: {len(docs)} posts")
    except Exception as e:
        print(f"❌ Blog error: {e}")
    return docs


# ── DOCX loader (unstructured) ────────────────────────────────

def _table_to_text(element) -> str:
    html = getattr(element.metadata, "text_as_html", None)
    if html:
        soup = BeautifulSoup(html, "html.parser")
        rows = []
        for tr in soup.find_all("tr"):
            cells = [
                td.get_text(separator=" ", strip=True)
                for td in tr.find_all(["td", "th"])
            ]
            if any(c for c in cells):
                rows.append(" | ".join(cells))
        if rows:
            return "\n".join(rows)
    return element.text.strip()


def _read_docx_unstructured(path: str) -> tuple[str, str, bool]:
    from unstructured.partition.docx import partition_docx

    elements = partition_docx(filename=path)
    parts, title = [], ""

    for el in elements:
        category = el.category
        if category not in CONTENT_ELEMENTS:
            continue

        if category == "Table":
            table_text = _table_to_text(el)
            if table_text:
                parts.append(f"\n[BẢNG]\n{table_text}\n[/BẢNG]\n")
        else:
            text = el.text.strip()
            if not text:
                continue
            if category in ("Title", "Header"):
                if not title:
                    title = text
                parts.append(f"\n## {text}\n")
            elif category == "ListItem":
                parts.append(f"• {text}")
            else:
                parts.append(text)

    content = "\n".join(parts).strip()
    content = _clean_content(content)
    content = _inject_headings(content)
    return content, title, _is_form(os.path.basename(path), content)


def _read_txt(path: str) -> tuple[str, str, bool]:
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    lines, title, parts = [l.strip() for l in raw.splitlines()], "", []
    for line in lines:
        if not line:
            continue
        m = re.match(r"^#{1,4}\s+(.+)", line)
        if m and not title:
            title = m.group(1)
        parts.append(line)
    return "\n".join(parts).strip(), title, False


def load_docx(folder: str = "data/docx") -> list[Document]:
    docs = []
    if not os.path.exists(folder):
        print(f"⚠️  Folder '{folder}' không tồn tại")
        return docs

    READERS = {".docx": _read_docx_unstructured, ".txt": _read_txt, ".md": _read_txt}
    files   = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in READERS]

    if not files:
        print(f"⚠️  Không có file trong '{folder}'")
        return docs

    for filename in sorted(files):
        path   = os.path.join(folder, filename)
        suffix = os.path.splitext(filename)[1].lower()
        print(f"📄 Đọc: {filename}")
        try:
            content, heading_title, is_form = READERS[suffix](path)
        except ImportError:
            print("❌ Thiếu thư viện: pip install 'unstructured[docx]'")
            continue
        except Exception as e:
            print(f"❌ Lỗi {filename}: {e}")
            continue

        if len(content) < 50:
            print(f"⚠️  Skip (quá ngắn): {filename}")
            continue

        title       = heading_title or re.sub(r"\.(docx|txt|md)$", "",
                          filename.replace("_", " "), flags=re.IGNORECASE).strip()
        search_text = title + "\n" + content[:500]
        doc_type    = _detect_doc_type(search_text, filename)
        doc_number  = _extract_doc_number(search_text)
        table_count = content.count("[BẢNG]")
        heading_count = content.count("\n## ")

        docs.append(Document(
            page_content=content,
            metadata={
                "title":         title,      "source":      "docx",
                "doc_type":      doc_type,   "filename":    filename,
                "doc_number":    doc_number, "issued_by":   _extract_issued_by(content),
                "is_form":       is_form,    "table_count": table_count,
                "heading_count": heading_count,
            },
        ))
        tag = "📋 BIỂU MẪU" if is_form else "✅"
        print(f"{tag} | type={doc_type} | tables={table_count} | headings={heading_count} | {len(content):,} chars")

    forms = sum(1 for d in docs if d.metadata.get("is_form"))
    print(f"\n✅ Tổng DOCX: {len(docs)} files ({forms} biểu mẫu)")
    return docs


# ── Chunker ───────────────────────────────────────────────────

def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter_law = RecursiveCharacterTextSplitter(
        chunk_size=3000, chunk_overlap=300,
        separators=["\nĐiều ", "\nChương ", "\nMục ", "\n## ", "\n[BẢNG]", "\n\n", "\n", ".", " "],
    )
    splitter_gen = RecursiveCharacterTextSplitter(
        chunk_size=3000, chunk_overlap=300,
        separators=["\n\n", "\n", ".", " "],
    )
    LAW_TYPES = {"luat","thong_tu","nghi_quyet","quyet_dinh","nghi_dinh","ket_luan","quy_dinh"}

    chunks = []
    for doc in docs:
        if doc.metadata.get("is_form"):
            chunks.append(doc)
            continue
        dt = doc.metadata.get("doc_type", "tai_lieu")
        sp = splitter_law if dt in LAW_TYPES else splitter_gen
        chunks.extend(sp.split_documents([doc]))

    filtered = [c for c in chunks if len(c.page_content.strip()) >= 80]
    forms   = sum(1 for c in filtered if c.metadata.get("is_form"))
    print(f"✅ Chunks: {len(filtered)} total ({len(filtered)-forms} content + {forms} biểu mẫu)")
    return filtered


# ── Entry point ───────────────────────────────────────────────

def load_all() -> list[Document]:
    print("📥 Loading blog...")
    blog_docs = load_blog()
    print("\n📥 Loading DOCX (unstructured)...")
    docx_docs = load_docx()
    all_docs  = blog_docs + docx_docs
    print(f"\n📄 Total: {len(all_docs)} docs (blog={len(blog_docs)}, docx={len(docx_docs)})")
    return all_docs


def save_raw(docs: list[Document], raw_dir: str = "data/raw") -> None:
    out = Path(raw_dir)
    if out.exists():
        shutil.rmtree(out)
        print(f"🗑️  Đã xoá {raw_dir}/ cũ")
    out.mkdir(parents=True)

    grouped: dict[str, dict] = {}
    for doc in docs:
        meta       = doc.metadata
        title      = meta.get("title", "unknown")
        article_id = re.sub(r"[^\w\-]", "_", title)[:60]
        is_form    = meta.get("is_form", False)
        category   = "bieu_mau" if is_form else meta.get("doc_type", "misc")

        if article_id not in grouped:
            grouped[article_id] = {
                "metadata": {
                    "article_id":  article_id,
                    "title":       title,
                    "source":      meta.get("source", ""),
                    "doc_type":    meta.get("doc_type", ""),
                    "category":    category,
                    "doc_number":  meta.get("doc_number", ""),
                    "issued_by":   meta.get("issued_by", ""),
                    "filename":    meta.get("filename", ""),
                    "url":         meta.get("url", ""),
                    "is_form":     is_form,
                    "table_count": meta.get("table_count", 0),
                },
                "plain_text": doc.page_content,
                "_category":  category,
            }
        else:
            grouped[article_id]["plain_text"] += "\n\n" + doc.page_content

    saved = 0
    for article_id, data in grouped.items():
        category = data.pop("_category")
        folder   = out / category
        folder.mkdir(parents=True, exist_ok=True)
        filepath = folder / f"{article_id}.json"
        filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tag = "📋" if data["metadata"].get("is_form") else "💾"
        print(f"  {tag} {filepath.name} ({len(data['plain_text']):,} chars)")
        saved += 1

    print(f"✅ Saved {saved} files → {raw_dir}/")


if __name__ == "__main__":
    print("=" * 50)
    print("📥 Loader — TNU-AIQA (unstructured)")
    print("=" * 50)
    docs = load_all()
    save_raw(docs)
    print("\n✅ Xong! Tiếp theo:")
    print("   python -m pipeline.kg_build.entity_extractor")
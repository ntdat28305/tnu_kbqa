"""
pipeline/ingestion/scraper.py
Scrape văn bản pháp luật / tin tức GDĐH từ discovered_urls.json

Nguồn hỗ trợ:
  - moet.gov.vn       (Bộ GD&ĐT)
  - chinhphu.vn       (Cổng TTĐT Chính phủ)
  - thuvienphapluat.vn
  - vnexpress.net / tuoitre.vn (tin tức GDĐH)

Chạy từ root project:
    python -m pipeline.ingestion.scraper --limit 5 --dry-run
    python -m pipeline.ingestion.scraper --limit 50
    python -m pipeline.ingestion.scraper
"""
from __future__ import annotations

import argparse
import json
import re
import time
import random
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from utils.logger import get_logger                          # ← SỬA

logger = get_logger(__name__, log_file="logs/scraper.log")  # ← GIỮ NGUYÊN

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
}

URL_FILE   = "data/discovered_urls.json"
OUTPUT_DIR = "data/raw"

# Mapping domain → loại nguồn (dùng để chọn parser)
DOMAIN_TYPE = {
    "moet.gov.vn":          "gov",
    "chinhphu.vn":          "gov",
    "thuvienphapluat.vn":   "phapluat",
    "vanban.chinhphu.vn":   "gov",
    "luatvietnam.vn":       "phapluat",
    "vnexpress.net":        "news",
    "tuoitre.vn":           "news",
    "dantri.com.vn":        "news",
    "nhandan.vn":           "news",
}


# ── Data classes ──────────────────────────────────────────────

@dataclass
class ArticleMetadata:
    url:         str
    article_id:  str
    title:       str
    category:    str          # ví dụ: "luat", "nghi_quyet", "tin_tuc"
    source_type: str          # "gov" | "phapluat" | "news"
    domain:      str
    doc_number:  str          # số văn bản nếu có, vd "125/2025/QH15"
    issued_by:   str          # cơ quan ban hành
    issued_date: str
    scraped_at:  str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RawArticle:
    metadata:           ArticleMetadata
    plain_text:         str
    headings:           list[str]
    articles_mentioned: list[str]   # các điều luật được đề cập, vd ["Điều 14", "Điều 32"]
    organizations:      list[str]   # tổ chức được đề cập
    doc_references:     list[str]   # số văn bản tham chiếu khác


# ── Load URLs ─────────────────────────────────────────────────

def load_urls(url_file: str = URL_FILE) -> dict[str, list[str]]:
    """Load discovered_urls.json → dict {category: [urls]}"""
    data = json.loads(Path(url_file).read_text(encoding="utf-8"))
    total = sum(len(v) for v in data.values())
    logger.info(f"Loaded {total} URLs từ {url_file}")
    return data


# ── Session ───────────────────────────────────────────────────

def create_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


# ── Scrape một article ────────────────────────────────────────

def scrape_article(
    session:  requests.Session,
    url:      str,
    category: str,
) -> RawArticle | None:
    time.sleep(random.uniform(1.5, 3.0))

    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
    except requests.RequestException as e:
        logger.warning(f"Lỗi fetch: {url} — {e}")
        return None

    soup   = BeautifulSoup(resp.text, "lxml")
    domain = _get_domain(url)
    stype  = DOMAIN_TYPE.get(domain, "unknown")

    title = _get_title(soup)
    if not title:
        logger.warning(f"Không tìm được title: {url}")
        return None

    content = _get_content(soup, stype)
    if not content:
        logger.warning(f"Không tìm được content: {url}")
        return None

    plain = _to_text(content)
    if len(plain) < 100:
        logger.warning(f"Content quá ngắn ({len(plain)} chars): {url}")
        return None

    return RawArticle(
        metadata=ArticleMetadata(
            url=url,
            article_id=_make_article_id(url, soup),
            title=title,
            category=category,
            source_type=stype,
            domain=domain,
            doc_number=_extract_doc_number(title + " " + plain[:300]),
            issued_by=_extract_issued_by(soup, plain[:500]),
            issued_date=_get_issued_date(soup),
        ),
        plain_text=plain,
        headings=_get_headings(content),
        articles_mentioned=_extract_legal_articles(plain),
        organizations=_extract_organizations(plain[:1000]),
        doc_references=_extract_doc_references(plain),
    )


# ── HTML helpers ──────────────────────────────────────────────

def _get_domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host.replace("www.", "")


def _get_title(soup: BeautifulSoup) -> str:
    for sel in ["h1.title", "h1.article-title", "h1"]:
        el = soup.select_one(sel)
        if el:
            return el.get_text(strip=True)
    tag = soup.find("title")
    if tag:
        return re.sub(r"\s*[|\-–]\s*.{3,40}$", "", tag.get_text(strip=True)).strip()
    return ""


def _get_content(soup: BeautifulSoup, source_type: str):
    """Chọn content element phù hợp với từng loại nguồn."""
    selectors = {
        "gov":      ["div.content-detail", "div#content", "div.van-ban-content",
                     "article", "div.noi-dung"],
        "phapluat": ["div.content1", "div#toanvan", "div.content-detail", "article"],
        "news":     ["article.fck_detail", "div.singular-content",
                     "div#mainContent", "article", "div.content-detail"],
    }
    for sel in selectors.get(source_type, ["main", "article"]):
        el = soup.select_one(sel) if isinstance(sel, str) else soup.find(sel)
        if el:
            return el

    # Fallback chung
    for sel in ["main", "article", {"role": "main"}]:
        el = soup.find(sel) if isinstance(sel, str) else soup.find(attrs=sel)
        if el:
            return el
    return None


def _to_text(el) -> str:
    for tag in el.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    lines = [ln.strip() for ln in el.get_text(separator="\n").splitlines() if ln.strip()]
    return "\n".join(lines)


def _get_headings(el) -> list[str]:
    return [
        f"{t.name.upper()}: {t.get_text(strip=True)}"
        for t in el.find_all(["h1", "h2", "h3", "h4"])
        if t.get_text(strip=True)
    ]


def _get_issued_date(soup: BeautifulSoup) -> str:
    tag = soup.find("time")
    if tag:
        return tag.get("datetime", tag.get_text(strip=True))
    meta = soup.find("meta", {"name": re.compile("date", re.I)})
    if meta:
        return meta.get("content", "")
    return ""


def _make_article_id(url: str, soup: BeautifulSoup) -> str:
    """Tạo ID duy nhất: ưu tiên số văn bản, fallback url path."""
    text = soup.get_text()[:500]
    m = re.search(r"(\d{1,4}/\d{4}/[A-ZĐ/-]{2,20})", text)
    if m:
        return re.sub(r"[/\\]", "_", m.group(1))
    return urlparse(url).path.rstrip("/").split("/")[-1][:60]


# ── Extractors chuyên biệt cho văn bản pháp luật ─────────────

def _extract_doc_number(text: str) -> str:
    """Trích số văn bản: 125/2025/QH15, 71-NQ/TW, v.v."""
    patterns = [
        r"\b\d{1,4}/\d{4}/(?:QH|NĐ|TT|QĐ|NQ|CT|BGDĐT|BGDDT|CP|TTg)[A-Z0-9/-]*",
        r"\b\d{1,4}-(?:NQ|CT|KL|TB)/TW\b",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(0)
    return ""


def _extract_issued_by(soup: BeautifulSoup, text: str) -> str:
    """Trích cơ quan ban hành từ đầu văn bản."""
    known = [
        "Quốc hội", "Chính phủ", "Thủ tướng Chính phủ",
        "Bộ Chính trị", "Ban Chấp hành Trung ương",
        "Bộ Giáo dục và Đào tạo", "Bộ Khoa học và Công nghệ",
        "Ủy ban nhân dân",
    ]
    text_lower = text.lower()
    for org in known:
        if org.lower() in text_lower:
            return org
    return ""


def _extract_legal_articles(text: str) -> list[str]:
    """Trích các điều luật được đề cập: Điều 14, khoản 3 Điều 5, v.v."""
    found = re.findall(r"(?:khoản\s+\d+\s+)?[Đđ]iều\s+\d+(?:\s+[A-Za-zÀ-ỹ]+)?", text)
    seen, results = set(), []
    for f in found:
        key = f.strip()
        if key not in seen:
            seen.add(key)
            results.append(key)
    return results[:30]


def _extract_organizations(text: str) -> list[str]:
    """Trích tên tổ chức từ đầu văn bản."""
    patterns = [
        r"(?:Trường|Đại học|Học viện|Viện)\s+[A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂẮẶ][^\n,;]{3,50}",
        r"(?:Bộ|Ủy ban|Tổng cục|Cục)\s+[A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂẮẶ][^\n,;]{3,40}",
    ]
    found = []
    for p in patterns:
        found += re.findall(p, text)
    seen, results = set(), []
    for f in found:
        key = f.strip()[:80]
        if key not in seen:
            seen.add(key)
            results.append(key)
    return results[:15]


def _extract_doc_references(text: str) -> list[str]:
    """Trích các số văn bản được tham chiếu trong nội dung."""
    pattern = r"\b\d{1,4}/\d{4}/[A-ZĐ][A-Z0-9/-]{1,15}"
    found   = list(dict.fromkeys(re.findall(pattern, text)))
    return found[:20]


# ── Lưu file ──────────────────────────────────────────────────

def save_article(article: RawArticle, output_dir: str = OUTPUT_DIR) -> Path:
    out = Path(output_dir) / article.metadata.category
    out.mkdir(parents=True, exist_ok=True)
    filepath = out / f"{article.metadata.article_id}.json"
    filepath.write_text(
        json.dumps(asdict(article), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return filepath


# ── Pipeline ──────────────────────────────────────────────────

def run(limit: int | None = None, dry_run: bool = False) -> None:
    url_data = load_urls()
    session  = create_session()
    saved, errors = 0, 0

    all_items = [
        (category, url)
        for category, urls in url_data.items()
        for url in urls
    ]
    if limit:
        all_items = all_items[:limit]

    total = len(all_items)
    logger.info(f"Bắt đầu scrape {total} articles...")

    for i, (category, url) in enumerate(all_items, 1):
        logger.info(f"[{i}/{total}] [{category}] {url}")

        article = scrape_article(session, url, category)
        if article is None:
            errors += 1
            continue

        if dry_run:
            logger.info(
                f"  [DRY RUN] '{article.metadata.title[:60]}' | "
                f"doc={article.metadata.doc_number} | "
                f"len={len(article.plain_text)} chars"
            )
        else:
            path = save_article(article)
            logger.info(f"  Saved → {path}")
            saved += 1

    logger.info(f"=== Xong: {saved} saved, {errors} errors ===")


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",   type=int, default=None, help="Giới hạn số article")
    parser.add_argument("--dry-run", action="store_true",    help="Chỉ log, không lưu")
    args = parser.parse_args()
    run(limit=args.limit, dry_run=args.dry_run)
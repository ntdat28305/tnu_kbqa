# TNU-AIQA KBQA v2

Hệ thống hỏi đáp kiểm định chất lượng giáo dục — Trường Đại học Tây Nguyên.

## Kiến trúc

```
Dữ liệu (DOCX + Blog)
       │
       ▼
pipeline/ingestion/loader.py     ← Load blog (RSS) + DOCX → data/raw/
       │
       ▼
pipeline/kg_build/entity_extractor.py  ← Extract entities bằng Groq LLM
       │
       ├──→ pipeline/kg_build/kg_builder.py   ← Build NetworkX KG → data/kg/
       └──→ pipeline/build_vector.py          ← Build ChromaDB    → data/chroma_db/
                     │
                     ▼
              app/rag_chain.py
              ┌─────────────────────────────────┐
              │  KG search (NetworkX)           │
              │  + Vector search (ChromaDB/e5)  │
              │  + BM25 search                  │
              │  → Rerank → LLM (Groq/Gemini)   │
              └─────────────────────────────────┘
                     │
                     ▼
              FastAPI /chat  ←→  Blogger widget
```

## Cài đặt

```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Linux/macOS

pip install -r requirements.txt
cp .env.example .env
# Điền GROQ_API_KEY_1 vào .env
```

## Pipeline (chạy 1 lần khi có dữ liệu mới)

```bash
# 1. Load blog + DOCX → data/raw/
python -m pipeline.ingestion.loader

# 2. Extract entities → data/processed/
python -m pipeline.kg_build.entity_extractor

# 3a. Build Knowledge Graph → data/kg/
python -m pipeline.kg_build.kg_builder

# 3b. Build Vector Store → data/chroma_db/
python -m pipeline.build_vector
```

## Chạy app

```bash
# Terminal 1 — API
uvicorn app.main:app --reload --port 8000

# Terminal 2 — UI
streamlit run app/streamlit_app.py
```

## Cấu hình

| Biến | Mô tả |
|------|-------|
| `GROQ_API_KEY_1` | Groq API key (bắt buộc) |
| `GROQ_API_KEY_2` | Key phụ để rotate (optional) |
| `GEMINI_API_KEY` | Gemini fallback (optional) |

## Thay đổi từ v1

- ✅ **Bỏ Neo4j** → dùng **NetworkX** (lightweight, không cần server)
- ✅ KG lưu file JSON → deploy Render/Railway không cần DB ngoài
- ✅ Thêm Blogger RSS scraper tự động (`feedparser`)
- ✅ Chunking theo Điều/Khoản cho văn bản pháp lý
- ✅ KG relation context được inject vào LLM prompt

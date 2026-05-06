import gc
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.schemas import ChatRequest, ChatResponse
from app.agent import run_agent
import psutil
import os

load_dotenv()

app = FastAPI(title="TNU-AIQA Chatbot API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok", "service": "TNU-AIQA KBQA v2 (Hybrid RAG + NetworkX KG)"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = run_agent(request.message)
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            model_used=result["model_used"]
        )
    finally:
        gc.collect()

@app.get("/status")
def status():
    from app.rag_chain import _current_combo_idx, _groq_combinations, GROQ_API_KEYS, GROQ_MODELS
    if _groq_combinations:
        api_key, model = _groq_combinations[_current_combo_idx]
        return {
            "status":             "ok",
            "current_key":        f"...{api_key[-6:]}",
            "current_model":      model,
            "combo_idx":          _current_combo_idx,
            "total_combinations": len(_groq_combinations),
        }
    return {"status": "no groq keys"}

@app.get("/metrics")
def metrics():
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    return {
        "ram_used_mb": round(mem.rss / 1024 / 1024, 1),
        "ram_vms_mb":  round(mem.vms / 1024 / 1024, 1),
        "cpu_percent": process.cpu_percent(),
    }

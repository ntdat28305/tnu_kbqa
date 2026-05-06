import gc
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from app.schemas import ChatRequest, ChatResponse
from app.agent import run_agent, run_agent_stream
import psutil
import os
import json

load_dotenv()

app = FastAPI(title="TNU-AIQA Chatbot API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok", "service": "TNU-AIQA KBQA v2.1 (Hybrid RAG + Streaming)"}

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

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """
    Streaming endpoint — trả về từng token ngay khi LLM generate.
    Format: Server-Sent Events (SSE)
    """
    def generate():
        try:
            for chunk in run_agent_stream(request.message):
                # SSE format: data: {...}\n\n
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        finally:
            gc.collect()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )

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
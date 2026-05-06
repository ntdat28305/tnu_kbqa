from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  # fix warning
    
    answer: str
    sources: list[str] = []
    model_used: str
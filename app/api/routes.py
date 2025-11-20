import os
import shutil
from typing import AsyncGenerator
from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from app.core.security import verify_user
from app.models.schemas import Pergunta
from app.services.agent import agent_executor
from app.services.rag_service import processar_pdf_interno

router = APIRouter()

# --- Lógica de Streaming ---
async def stream_agent(question: str) -> AsyncGenerator[str, None]:
    async for chunk in agent_executor.astream_events(
        {"messages": [HumanMessage(content=question)]},
        version="v1"
    ):
        if chunk["event"] == "on_chat_model_stream":
            if chunk["data"]["chunk"].content: 
                yield chunk["data"]["chunk"].content

# --- Rotas ---

@router.get("/")
async def root():
    return {"message": "Zaia Agent Modular Online"}

@router.post("/chat")
async def chat(p: Pergunta, user: str = Depends(verify_user)):
    return StreamingResponse(stream_agent(p.question), media_type="text/plain")

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), user: str = Depends(verify_user)):
    try:
        os.makedirs("uploads", exist_ok=True)
        loc = f"uploads/{file.filename}"
        with open(loc, "wb+") as buffer: 
            shutil.copyfileobj(file.file, buffer)
        return {"status": processar_pdf_interno(loc)}
    except Exception as e: 
        return {"status": str(e)}
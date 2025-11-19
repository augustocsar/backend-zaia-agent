# main.py
import os
import shutil
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import AsyncGenerator

# --- Importações do LangChain ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- RAG LOCAL (Sem API, Sem Cota) ---
from langchain_huggingface import HuggingFaceEmbeddings
# -------------------------------------

# --- Configuração ---
load_dotenv()

HG_API_KEY = os.getenv("HG_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not HG_API_KEY or not GOOGLE_API_KEY:
    raise EnvironmentError("Faltam chaves! Verifique HG_API_KEY e GOOGLE_API_KEY no .env")

# --- App ---
app = FastAPI(title="Zaia Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Auth ---
security = HTTPBasic()
def verify(creds: HTTPBasicCredentials = Depends(security)):
    if creds.username != "admin" or creds.password != "1234":
        raise HTTPException(401, "Credenciais inválidas")
    return creds.username

# --- Ferramentas ---
@tool
def get_clima(cidade: str) -> str:
    """Obtém clima."""
    try:
        url = f"https://api.hgbrasil.com/weather?key={HG_API_KEY}&city_name={cidade}"
        r = requests.get(url).json()["results"]
        return f"Clima em {r['city']}: {r['temp']}C, {r['description']}."
    except: return "Erro ao buscar clima."

@tool
def get_cotacao(moeda: str = "dólar") -> str:
    """Obtém cotação."""
    try:
        url = f"https://api.hgbrasil.com/finance?key={HG_API_KEY}"
        d = requests.get(url).json()["results"]["currencies"]
        code = "USD" if "dolar" in moeda.lower() else "EUR" if "euro" in moeda.lower() else "BTC"
        if code in d: return f"{d[code]['name']}: R$ {d[code]['buy']}"
    except: pass
    return "Cotação não encontrada."

# --- RAG LOCAL ---
vectorstore = None

def processar_pdf_interno(caminho: str) -> str:
    global vectorstore
    if not os.path.exists(caminho): return "Arquivo não existe."
    
    try:
        # Carrega
        print("Carregando PDF...")
        loader = PyPDFLoader(caminho)
        docs = loader.load()
        
        # Divide
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        # Indexa LOCALMENTE (Hugging Face na CPU)
        print("Criando Embeddings Locais...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Salva
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return "PDF processado com sucesso (Local)!"
        
    except Exception as e:
        print(f"ERRO RAG: {e}")
        return f"Erro interno: {str(e)}"

@tool
def carregar_pdf(caminho: str) -> str:
    """Carrega PDF."""
    return processar_pdf_interno(caminho)

@tool
def buscar_no_pdf(pergunta: str) -> str:
    """
    Use esta ferramenta para responder QUALQUER pergunta sobre o conteúdo do documento ou PDF anexado.
    O PDF JÁ ESTÁ CARREGADO na memória.
    Não peça para o usuário carregar o arquivo.
    Apenas pesquise a resposta aqui.
    """
    global vectorstore
    if not vectorstore: return "Nenhum PDF carregado."
    try:
        docs = vectorstore.similarity_search(pergunta, k=3)
        return "\n".join([d.page_content[:500] for d in docs])
    except: return "Erro na busca."

tools = [get_clima, get_cotacao, buscar_no_pdf]

# --- AGENTE (Aqui está o Gemini 2.0 Flash) ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", # <--- AGORA SIM!
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    streaming=True
)
agent = create_react_agent(llm, tools)

class Pergunta(BaseModel): question: str

async def stream_agent(question: str) -> AsyncGenerator[str, None]:
    async for chunk in agent.astream_events({"messages": [HumanMessage(content=question)]}, version="v1"):
        if chunk["event"] == "on_chat_model_stream":
            if chunk["data"]["chunk"].content: yield chunk["data"]["chunk"].content

@app.post("/chat")
async def chat(p: Pergunta, user: str = Depends(verify)):
    return StreamingResponse(stream_agent(p.question), media_type="text/plain")

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), user: str = Depends(verify)):
    try:
        os.makedirs("uploads", exist_ok=True)
        loc = f"uploads/{file.filename}"
        with open(loc, "wb+") as buffer: shutil.copyfileobj(file.file, buffer)
        return {"status": processar_pdf_interno(loc)}
    except Exception as e: return {"status": str(e)}

# uvicorn main:app --reload
# main.py
import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware  # <--- ADICIONADO
from pydantic import BaseModel
from typing import AsyncGenerator

# LangChain + LangGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

HG_API_KEY = os.getenv("HG_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not HG_API_KEY or not GOOGLE_API_KEY:
    raise EnvironmentError("Chaves de API não encontradas no .env")

# ---------- FastAPI com CORS ----------
app = FastAPI(
    title="Zaia Agent",
    description="Agente com clima, dólar, PDF e streaming",
    version="1.0.0"
)

# LIBERA CORS PARA O FRONTEND (Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# ---------- Autenticação Mock ----------
security = HTTPBasic()

def verify(creds: HTTPBasicCredentials = Depends(security)):
    if creds.username != "admin" or creds.password != "1234":
        raise HTTPException(401, "Credenciais inválidas")
    return creds.username

# ---------- Ferramentas ----------
@tool
def get_clima(cidade: str) -> str:
    """Obtém o clima de uma cidade."""
    url = f"https://api.hgbrasil.com/weather?key={HG_API_KEY}&city_name={cidade}"
    try:
        r = requests.get(url, timeout=10).json()["results"]
        return f"Clima em {r['city']}: {r['temp']}°C, {r['description'].lower()}."
    except:
        return "Erro: cidade não encontrada."

@tool
def get_cotacao(moeda: str = "dólar") -> str:
    """Obtém a cotação de qualquer moeda disponível na HG Brasil.
    Exemplos: dólar, euro, bitcoin, libra, iene, etc."""
    moeda = moeda.lower().strip()
    
    # Mapeamento de nomes comuns → código da API
    mapeamento = {
        "dolar": "USD", "dólar": "USD", "usd": "USD",
        "euro": "EUR", "€": "EUR",
        "bitcoin": "BTC", "btc": "BTC",
        "libra": "GBP", "libra esterlina": "GBP",
        "iene": "JPY", "iêne": "JPY", "yen": "JPY",
        "dolar australiano": "AUD", "aud": "AUD",
        "dolar canadense": "CAD", "cad": "CAD",
        "franco suíço": "CHF", "chf": "CHF",
    }
    
    codigo = mapeamento.get(moeda)
    if not codigo:
        # Tenta buscar diretamente pelo código (ex: usuário digita "EUR")
        if moeda.upper() in ["USD", "EUR", "BTC", "GBP", "JPY", "AUD", "CAD", "CHF"]:
            codigo = moeda.upper()
        else:
            return f"Desculpe, não tenho cotação para '{moeda}'. Tente: dólar, euro, bitcoin, libra, iene..."
    
    url = f"https://api.hgbrasil.com/finance?key={HG_API_KEY}"
    try:
        data = requests.get(url, timeout=10).json()["results"]["currencies"]
        moeda_info = data.get(codigo)
        if not moeda_info or "buy" not in moeda_info:
            return f"Não consegui obter a cotação do {moeda} no momento."
        
        nome = moeda_info.get("name", codigo)
        valor = moeda_info["buy"]
        variacao = moeda_info.get("variation", 0)
        
        return f"A cotação do {nome} é R$ {valor:.4f} (variação: {variacao}%)"
        
    except Exception as e:
        return "Erro ao consultar cotação. Tente novamente."
# RAG para PDF
vectorstore = None

@tool
def carregar_pdf(caminho: str) -> str:
    """Carrega um PDF para busca."""
    global vectorstore
    if not os.path.exists(caminho):
        return "PDF não encontrado."
    loader = PyPDFLoader(caminho)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return "PDF carregado com sucesso."

@tool
def buscar_no_pdf(pergunta: str) -> str:
    """Busca no PDF carregado."""
    if vectorstore is None:
        return "Nenhum PDF carregado."
    docs = vectorstore.similarity_search(pergunta, k=3)
    return "\n\n".join([d.page_content[:500] for d in docs])

tools = [get_clima, get_cotacao, carregar_pdf, buscar_no_pdf]

# ---------- LLM ----------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # ← CORRIGIDO
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    streaming=True,
)

# ---------- Agente com LangGraph ----------
agent = create_react_agent(llm, tools)

# ---------- Modelos ----------
class Pergunta(BaseModel):
    question: str

async def stream_agent(question: str) -> AsyncGenerator[str, None]:
    async for chunk in agent.astream_events(
        {"messages": [HumanMessage(content=question)]},
        version="v1"
    ):
        if chunk["event"] == "on_chat_model_stream":
            content = chunk["data"]["chunk"].content
            # REMOVE TODOS OS "1" E ESPAÇOS
            clean_content = content.replace("1", "").strip()
            if clean_content:
                yield clean_content

# ---------- Endpoints ----------
@app.get("/")
async def root():
    return {"message": "Zaia Agent - Login: admin/1234"}

@app.post("/chat")
async def chat(p: Pergunta, user: str = Depends(verify)):
    return StreamingResponse(stream_agent(p.question), media_type="text/plain")

@app.post("/upload-pdf")
async def upload_pdf(req: Request, user: str = Depends(verify)):
    data = await req.json()
    caminho = data.get("path", "docs/documento.pdf")
    return {"status": carregar_pdf(caminho)}
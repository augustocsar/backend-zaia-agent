# main.py
import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from typing import AsyncGenerator

# LangChain + LangGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
def get_dolar() -> str:
    """Obtém cotação do dólar."""
    url = f"https://api.hgbrasil.com/finance?key={HG_API_KEY}"
    try:
        r = requests.get(url, timeout=10).json()["results"]["currencies"]["USD"]
        return f"Dólar: R$ {r['buy']} (variação: {r['variation']}%)"
    except:
        return "Erro ao obter dólar."

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

tools = [get_clima, get_dolar, carregar_pdf, buscar_no_pdf]

# ---------- LLM ----------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    streaming=True,
)

# ---------- Agente com LangGraph ----------
agent = create_react_agent(llm, tools)

# ---------- FastAPI ----------
app = FastAPI()

class Pergunta(BaseModel):
    question: str

async def stream_agent(question: str) -> AsyncGenerator[str, None]:
    async for chunk in agent.astream_events(
        {"messages": [HumanMessage(content=question)]},
        version="v1"
    ):
        if chunk["event"] == "on_chain_stream" and "chunk" in chunk["data"]:
            yield chunk["data"]["chunk"].content

@app.post("/chat")
async def chat(p: Pergunta, user: str = Depends(verify)):
    return StreamingResponse(stream_agent(p.question), media_type="text/plain")

@app.post("/upload-pdf")
async def upload(req: Request, user: str = Depends(verify)):
    data = await req.json()
    return {"status": carregar_pdf(data.get("path", "documento.pdf"))}

@app.get("/")
async def root():
    return {"message": "Zaia Agent - Login: admin/1234"}
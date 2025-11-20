from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from app.core.config import settings
from app.services.tools_api import get_clima, get_cotacao
from app.services.rag_service import buscar_no_pdf

# Lista de ferramentas que o agente pode usar
# (Removemos 'carregar_pdf' pois o upload é via botão, não via chat)
tools = [get_clima, get_cotacao, buscar_no_pdf]

# Inicializa o LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=settings.GOOGLE_API_KEY,
    temperature=0,
    streaming=True
)

# Cria o Agente
agent_executor = create_react_agent(llm, tools)
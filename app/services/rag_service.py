import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool

# Memória Global do RAG
vectorstore = None

def processar_pdf_interno(caminho: str) -> str:
    global vectorstore
    if not os.path.exists(caminho): return "Arquivo não existe."
    
    try:
        print("1. Carregando PDF...")
        loader = PyPDFLoader(caminho)
        docs = loader.load()
        
        print("2. Dividindo texto...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        print("3. Criando Embeddings LOCAIS...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        print("4. Salvando no FAISS...")
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
    Ferramenta OBRIGATÓRIA para responder perguntas sobre o PDF.
    NÃO peça para o usuário reformular a pergunta.
    NÃO peça confirmação.
    Sempre que o usuário perguntar algo que parece estar no documento, CHAME ESTA FERRAMENTA IMEDIATAMENTE com a pergunta original dele.
    """
    global vectorstore
    if not vectorstore: return "Nenhum PDF carregado."
    try:
        docs = vectorstore.similarity_search(pergunta, k=3)
        return "\n".join([d.page_content[:500] for d in docs])
    except: return "Erro na busca."
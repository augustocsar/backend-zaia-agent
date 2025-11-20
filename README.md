# 🧠 Zaia Agent - Backend

API de Agente Inteligente construída com **FastAPI**, **LangChain** e **Gemini 2.0**.
Este sistema utiliza uma arquitetura modular para orquestrar ferramentas de clima, cotação financeira e RAG (Retrieval-Augmented Generation) para leitura de PDFs locais.

## 🚀 Tecnologias Utilizadas

* **Python 3.12+**
* **FastAPI:** Framework web moderno e rápido.
* **LangChain:** Orquestração do agente ReAct.
* **Google Gemini 2.0 Flash:** Cérebro do agente (LLM).
* **Hugging Face (Local):** Embeddings para RAG (`all-MiniLM-L6-v2`) rodando na CPU.
* **FAISS:** Banco vetorial em memória para busca semântica rápida.
* **HGBrasil:** APIs externas para dados em tempo real.

## 📂 Arquitetura

O projeto segue uma estrutura **MVC Adaptada** para serviços:

```text
backend-zaia-agent/
├── app/
│   ├── api/          # Rotas e Endpoints
│   ├── core/         # Configurações e Segurança (Auth)
│   ├── models/       # Schemas de dados (Pydantic)
│   ├── services/     # Lógica de Negócio (Agente, RAG, Tools)
│   └── __init__.py
├── .env              # Variáveis de ambiente (Chaves de API)
├── .gitignore        # Arquivos ignorados pelo Git
├── main.py           # Ponto de entrada da aplicação
└── requirements.txt  # Lista de dependências
```


## ⚙️ Configuração e Instalação

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/augustocsar/backend-zaia-agent.git](https://github.com/augustocsar/backend-zaia-agent.git)
    cd backend-zaia-agent
    ```

2.  **Crie o ambiente virtual e instale as dependências:**
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate
    
    pip install -r requirements.txt
    ```

3.  **Configure as Variáveis de Ambiente:**
    Crie um arquivo `.env` na raiz com suas chaves:
    ```ini
    HG_API_KEY="sua_chave_hg_brasil"
    GOOGLE_API_KEY="sua_chave_gemini"
    ```

4.  **Execute o servidor:**
    ```bash
    uvicorn main:app --reload
    ```

## 🔌 Endpoints Principais

* `POST /chat`: Envia perguntas para o agente (suporta Streaming).
* `POST /upload-pdf`: Recebe arquivos PDF para indexação vetorial local.

---
**Desenvolvido por Augusto César**
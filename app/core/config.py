import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    HG_API_KEY = os.getenv("HG_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Validação inicial
    if not HG_API_KEY or not GOOGLE_API_KEY:
        raise EnvironmentError("Faltam chaves no .env! Verifique HG e GOOGLE.")

settings = Settings()
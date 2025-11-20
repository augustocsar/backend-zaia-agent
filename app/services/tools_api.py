import requests
from langchain.tools import tool
from app.core.config import settings

@tool
def get_clima(cidade: str) -> str:
    """Obtém o clima de uma cidade."""
    try:
        url = f"https://api.hgbrasil.com/weather?key={settings.HG_API_KEY}&city_name={cidade}"
        r = requests.get(url).json()["results"]
        return f"Clima em {r['city']}: {r['temp']}°C, {r['description']}."
    except: return "Erro ao buscar clima."

@tool
def get_cotacao(moeda: str = "dólar") -> str:
    """Obtém a cotação de moedas."""
    try:
        url = f"https://api.hgbrasil.com/finance?key={settings.HG_API_KEY}"
        d = requests.get(url).json()["results"]["currencies"]
        code = "USD" if "dolar" in moeda.lower() else "EUR" if "euro" in moeda.lower() else "BTC"
        if code in d: return f"{d[code]['name']}: R$ {d[code]['buy']}"
    except: pass
    return "Cotação não encontrada."
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

def verify_user(creds: HTTPBasicCredentials = Depends(security)):
    if creds.username != "admin" or creds.password != "1234":
        raise HTTPException(401, "Credenciais inválidas")
    return creds.username
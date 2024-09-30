from fastapi import FastAPI

from app.routes import router

app = FastAPI()

@app.get("/ping")
def pong():
    return {"ping": "pong!"}

app.include_router(router, prefix="/rag-api")
from fastapi import APIRouter

from app.components import create_chunks, fetch_index, load_documents
from app.services import generate_response_based_on_docs

router = APIRouter()


@router.get("/prompt", tags=["rag-api"])
async def ask_prompt(prompt: str):
    return generate_response_based_on_docs(prompt)


@router.get("/documents", tags=["components"])
async def get_documents():
    return load_documents()


@router.get("/nodes", tags=["components"])
async def get_nodes():
    documents = load_documents()
    return create_chunks(documents)


@router.get("/index", tags=["components"])
async def get_index():
    return fetch_index()

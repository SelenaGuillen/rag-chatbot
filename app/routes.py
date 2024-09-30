from fastapi import APIRouter

from app.services import generate_response_based_on_docs
from app.components import create_chunks, load_documents, store_chunks

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


@router.get("/docstore", tags=["components"])
async def return_docstore():
    return store_chunks(create_chunks(load_documents()))
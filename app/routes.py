from fastapi import APIRouter

from app.components import create_chunks, load_documents, create_embeddings
from app.services import generate_response_based_on_docs
from app.util import convert_docs_to_data
from app.vector_db import populate_db

router = APIRouter()


@router.get("/prompt", tags=["rag-api"])
async def ask_prompt(prompt: str):
    return generate_response_based_on_docs(prompt)


@router.get("/documents", tags=["components"])
async def get_documents():
    return load_documents()


@router.get("/chunks", tags=["components"])
async def get_chunks():
    documents = load_documents()
    return create_chunks(documents)

@router.get("/embeddings", tags=["components"])
async def get_embeddings():
    documents = load_documents()
    chunks = create_chunks(documents)
    return create_embeddings(chunks)

@router.get("/get_vector_db", tags=["pinecone"])
async def get_vector_db():
    documents = load_documents()
    chunks = create_chunks(documents)
    embeddings = create_embeddings(chunks)
    return await populate_db(embeddings=embeddings)

@router.get("/docs_to_data", tags=["util"])
async def get_docs_to_data():
    return convert_docs_to_data(load_documents())



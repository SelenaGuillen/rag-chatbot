from fastapi import APIRouter

from app.components import (
    create_chunks,
    create_embedding_from_user_query,
    create_embeddings_from_chunks,
    load_documents,
)
from app.services import generate_response_based_on_docs
from app.util import convert_to_documents
from app.vector_db import populate_db, query_for_most_relevant, rerank_chunks

router = APIRouter()


@router.get("/prompt", tags=["rag-api"])
async def ask_prompt(prompt: str):
    return await generate_response_based_on_docs(prompt)


@router.get("/documents", tags=["components"])
async def get_documents():
    return load_documents()


@router.get("/chunks", tags=["components"])
async def get_chunks():
    documents = load_documents()
    return create_chunks(documents)


@router.get("/chunk_embeddings", tags=["components"])
async def get_embeddings():
    documents = load_documents()
    chunks = create_chunks(documents)
    return create_embeddings_from_chunks(chunks)


@router.get("/user_query_embedding", tags=["components"])
async def get_user_query_embedding(query: str):
    return create_embedding_from_user_query(query)


@router.get("/populate_vector_db", tags=["pinecone"])
async def populate_vector_db():
    documents = load_documents()
    chunks = create_chunks(documents)
    embeddings = create_embeddings_from_chunks(chunks)
    return await populate_db(embeddings)


@router.get("/query_vector_db", tags=["pinecone"])
async def query_vector_db(query: str):
    return await query_for_most_relevant(query)


@router.get("/top_chunks_after_rerank", tags=["components"])
async def get_top_chunks_after_rerank(query: str):
    matches = await query_for_most_relevant(query)
    return await rerank_chunks(matches, query)


@router.get("/docs_to_data", tags=["util"])
async def get_docs_to_data(query: str):
    reranked_chunks = await get_top_chunks_after_rerank(query)
    return await convert_to_documents(reranked_chunks)

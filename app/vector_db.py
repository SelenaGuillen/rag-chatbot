import logging
import os

from dotenv import load_dotenv
from pinecone import Pinecone

from app.client import RERANK_MODEL
from app.components import (create_embedding_from_user_query,
                            format_embeddings_for_vector_db)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("embed-english-v3")


# TODO: Use pydantic?
async def populate_db(embeddings_and_metadata: tuple) -> None:
    """
    Get the Pinecone vector database.
    """
    embeddings, text = embeddings_and_metadata
    vector_data = format_embeddings_for_vector_db(embeddings, text)
    # Initialized with 1024 dimensions, indexes using cosine similarity metric
    index.upsert(vector_data)
    return f"Populated Pinecone index with {len(vector_data)} vectors."


async def query_for_most_relevant(user_query: str) -> dict:
    """
    Query the vector db for the most relevant chunk based on the user query embedding.
    """
    user_query_embedding = create_embedding_from_user_query(user_query)
    matches = index.query(
        vector=user_query_embedding, top_k=3, include_values=True, include_metadata=True
    )
    return matches.to_dict()


async def rerank_chunks(matches: dict, query: str):
    """
    Rerank the chunks based on the matches from the vector db.
    """
    documents = []
    for match in matches["matches"]:
        document = {"id": match["id"], "text": match["metadata"].get("text")}
        documents.append(document)

    result = pc.inference.rerank(
        query=query,
        documents=documents,
        top_n=2,
        return_documents=True,
        model=RERANK_MODEL,
    )

    return result.to_dict()

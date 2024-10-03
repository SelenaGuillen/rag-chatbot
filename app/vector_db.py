from pinecone import Pinecone
import os
from dotenv import load_dotenv
import logging
from typing import List
from app.components import create_embedding_from_user_query
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("embed-english-v3")


async def populate_db(embeddings: list[float]) -> None:
    """
    Get the Pinecone vector database.
    """
    # Initialized with 1024 dimensions, indexes using cosine similarity metric
    list_of_vectors = []
    # Prepare vector data before upserting
    for i, embedding in enumerate(embeddings):
        list_of_vectors.append({"id": str(i), "values": embedding})
    logger.info(f"Populating Pinecone index with {len(list_of_vectors)} vectors.")
    index.upsert(list_of_vectors)
    return f"Populated Pinecone index with {len(list_of_vectors)} vectors."


async def query_for_most_relevant(user_query: str):
    """
    Query the vector db for the most relevant chunk based on the user query embedding.
    """
    user_query_embedding = create_embedding_from_user_query(user_query)
    matches = index.query(
        vector=user_query_embedding,
        top_k=3,
        include_values=True,
    )
    return json.dumps(matches.to_dict())
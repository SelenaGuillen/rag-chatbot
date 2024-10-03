from pinecone import Pinecone
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

async def populate_db(embeddings: list[float]) -> None:
    """
    Get the Pinecone vector database.
    """
    # Initialized with 1024 dimensions, indexes using cosine similarity metric
    index = pc.Index("embed-english-v3")
    
    list_of_vectors = []
    # Prepare vector data before upserting
    for i, embedding in enumerate(embeddings):
        list_of_vectors.append({"id": str(i), "values": embedding})
    logger.info(f"Populating Pinecone index with {len(list_of_vectors)} vectors.")
    index.upsert(list_of_vectors)
    return f"Populated Pinecone index with {len(list_of_vectors)} vectors."


    
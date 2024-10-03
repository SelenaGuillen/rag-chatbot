from typing import List

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode
import logging
from app.client import co, embed_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_documents() -> List[Document]:
    """
    Load documents from the data directory. Can be scaled up when using multiple workers.

    Returns:
    List[Document]: The documents loaded from the data directory.
    """
    docs = SimpleDirectoryReader("data").load_data()
    logger.info(f"Loaded {len(docs)} documents from the data directory.")
    return docs


def create_chunks(docs: List[Document]) -> List[TextNode]:
    """
    Splits up the documents into chunks with preference for complete sentences.
    SentenceSplitter is less likely to output hanging sentences or parts of sentences at the end of a chunk.
    """
    # Can experiment with chunk size and overlap once we have vector store index
    # data = convert_docs_to_data(docs)
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=0)
    chunks = splitter.get_nodes_from_documents(docs)
    logger.info(f"Created {len(chunks)} chunks from the documents.")
    return chunks

# TODO: Could try adding id prefixes to differentiate which chunk belongs to which document
def create_embeddings_from_chunks(chunks: List[TextNode]) -> List[List[float]]:
    """
    Create embeddings from the chunks using the cohere embed model from client.
    """
    # search_document to inform model should search chunks
    # returns type EmbedByTypeResponse 
    embeddings_response = co.embed(
        texts=[chunk.text for chunk in chunks],
        model=embed_model.model_name,
        input_type="search_document",
        embedding_types=['float']
    )
    embeddings = embeddings_response.embeddings.float
    logger.info(f"Created {len(embeddings)} embeddings from chunks.")
    return embeddings

def create_embedding_from_user_query(query: str) -> List[float]:
    """
    Create an embedding from the user query using the cohere embed model from client.
    """
    # search_query to query vector db
    # returns type EmbedByTypeResponse 
    embeddings_response = co.embed(
        texts=[query],
        model=embed_model.model_name,
        input_type="search_query",
        embedding_types=['float']
    )
    query_embedding = embeddings_response.embeddings.float[0]
    logger.info(f"Created an embedding from the user query.")
    return query_embedding

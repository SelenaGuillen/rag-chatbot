from typing import List

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode


def load_documents() -> List[Document]:
    """
    Load documents from the data directory. Can be scaled up when using multiple workers.

    Returns:
    List[Document]: The documents loaded from the data directory.
    """
    docs = SimpleDirectoryReader("data").load_data()
    return docs


def create_chunks(docs: List[Document]) -> List[TextNode]:
    """
    Splits up the documents into chunks with preference for complete sentences.
    SentenceSplitter is less likely to output hanging sentences or parts of sentences at the end of a chunk.
    """
    splitter = SentenceSplitter(chunk_size=500)
    nodes = splitter.get_nodes_from_documents(docs)
    return nodes


def fetch_index() -> VectorStoreIndex:
    """
    Get the index for the documents using embed model from Settings.
    """
    chunks = create_chunks(load_documents())
    index = VectorStoreIndex(chunks)
    return index

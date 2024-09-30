from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore


from typing import List


def load_documents() -> str:
    docs = SimpleDirectoryReader("data").load_data()
    return docs

def create_chunks(docs: str) -> List[TextNode]:
    splitter = SentenceSplitter(chunk_size=1000)
    nodes = splitter.get_nodes_from_documents(docs)
    return nodes

def store_chunks(nodes: List[TextNode]) -> SimpleDocumentStore:
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    return docstore

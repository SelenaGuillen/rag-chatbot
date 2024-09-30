from llama_index.core import SimpleDirectoryReader


def load_documents() -> str:
    docs = SimpleDirectoryReader("data").load_data()
    return docs
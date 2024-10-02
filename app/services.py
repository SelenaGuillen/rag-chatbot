import json

from llama_index.core.retrievers import VectorIndexAutoRetriever

from app.client import co
from app.components import load_documents
from app.util import convert_docs_to_data


# TODO: Replace with pydantic or models from llama_index
def generate_response_based_on_docs(user_prompt: str) -> str:
    """
    Generate a response based on the user prompt and the documents loaded from the data directory.

    Args:
    user_prompt (str): The prompt to generate a response from.

    Returns:
    str: The response generated based on the prompt and the
    """
    # Loading in all documents vs chunking/indexing/flitering
    docs = load_documents()
    data = convert_docs_to_data(docs)

    # VectorStoreIndex
    # index = fetch_index()

    prompt = f"Answer based on the following prompt using document context: {user_prompt}. Otherwise, state 'I cannot generate a response based on this information'."

    response = co.chat(
        documents=data,
        model="command-r",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response

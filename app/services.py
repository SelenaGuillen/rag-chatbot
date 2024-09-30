from app.client import co
from app.components import load_documents


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
    doc_text = ""
    for doc in docs:
        doc_text += doc.text

    # TODO: Replace all docs loaded into context with index
    # index = fetch_index()

    prompt = f"Based on the following prompt: {user_prompt}, use only this information: {doc_text} to generate a response. Otherwise, state 'I cannot generate a response based on this information'."
    response = co.chat(
        message=prompt,
        temperature=0,
    )
    return response

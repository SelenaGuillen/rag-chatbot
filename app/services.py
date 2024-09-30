from app.client import co
from app.util import load_documents


def generate_response_based_on_docs(user_prompt: str) -> str:
    docs = load_documents()
    doc_text = ""
    for doc in docs:
        doc_text += doc.text
    prompt = f"Based on the following prompt: {user_prompt}, use only this information: {docs} to generate a response. Otherwise, state 'I cannot generate a response based on this information'."
    response = co.chat(
        message=prompt,
        temperature=1.0,
    )
    return response

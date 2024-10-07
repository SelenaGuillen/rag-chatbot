from app.client import co
from app.util import convert_to_documents
from app.vector_db import query_for_most_relevant, rerank_chunks


# TODO: Replace with pydantic or models from llama_index
async def generate_response_based_on_docs(user_prompt: str) -> str:
    """
    Generate a response based on the user prompt and the documents loaded from the data directory.

    Args:
    user_prompt (str): The prompt to generate a response from.

    Returns:
    str: The response generated based on the prompt and the
    """
    matches = await query_for_most_relevant(user_prompt)
    reranked_data = await rerank_chunks(matches, user_prompt)
    data = await convert_to_documents(reranked_data)

    prompt = f"Only use documents as a source. Answer the prompt: {user_prompt}. If no answer is found, please say 'I cannot answer that with my current knowledge'."
    response = co.chat(
        # max limit is 125k for context/input and 4k for output
        max_tokens=2000,
        documents=data,
        model="command-r",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response

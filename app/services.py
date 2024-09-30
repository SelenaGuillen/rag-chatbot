from app.client import co

def generate_response(query: str):
    response = co.chat(
        message=query,
        model="command-r"
    )
    return response
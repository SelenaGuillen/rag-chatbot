from fastapi import APIRouter

from app.services import generate_response_based_on_docs

router = APIRouter()

@router.post("/prompt")
async def ask_prompt(prompt: str):
    return generate_response_based_on_docs(prompt)

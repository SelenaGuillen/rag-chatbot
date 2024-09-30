from fastapi import APIRouter
from app.services import generate_response

router = APIRouter()

@router.get("/query")
def submit_query(query: str):
    return generate_response(query)
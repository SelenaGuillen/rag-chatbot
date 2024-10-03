import os

import cohere
from dotenv import load_dotenv

load_dotenv()

CO_API_KEY = os.getenv("CO_API_KEY")

co = cohere.ClientV2(CO_API_KEY)

EMBED_MODEL = "embed-english-v3.0"
RERANK_MODEL = "bge-reranker-v2-m3"

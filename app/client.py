import os

import cohere
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.cohere import CohereEmbedding

load_dotenv()

CO_API_KEY = os.getenv("CO_API_KEY")

co = cohere.ClientV2(CO_API_KEY)

embed_model = CohereEmbedding(
    cohere_api_key=CO_API_KEY,
    model_name="embed-english-v3.0",
    input_type="search_query",
)


Settings.embed_model = embed_model

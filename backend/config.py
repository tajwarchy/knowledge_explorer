import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")


EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "llama-3.1-8b-instant"

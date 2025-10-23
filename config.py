import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_PATH = "chroma_storage"

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "llama-3.1-8b-instant"

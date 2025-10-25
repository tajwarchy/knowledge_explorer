from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .config import QDRANT_URL, QDRANT_API_KEY, EMBED_MODEL

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
vectorstore = Qdrant(client=client, collection_name="knowledge", embeddings=embeddings)


def ingest_text(text: str, metadata: dict = None):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = [
        Document(page_content=chunk, metadata=metadata or {})
        for chunk in splitter.split_text(text)
    ]
    vectorstore.add_documents(docs)


def retrieve_context(query: str, top_k=4):
    return vectorstore.similarity_search(query, k=top_k)

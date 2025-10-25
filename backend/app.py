from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os

from .document_utils import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_html,
    extract_text_from_txt,
)
from .vectorstore_utils import ingest_text, retrieve_context
from .llm_utils import call_groq_chat

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile):
    ext = file.filename.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    if ext == "pdf":
        text = extract_text_from_pdf(tmp_path)
    elif ext == "docx":
        text = extract_text_from_docx(tmp_path)
    elif ext == "html":
        text = extract_text_from_html(tmp_path)
    elif ext == "txt":
        text = extract_text_from_txt(tmp_path)
    else:
        return {"error": "Unsupported file type"}

    ingest_text(text, metadata={"source": file.filename})
    os.remove(tmp_path)
    return {"message": f"{file.filename} indexed successfully"}


@app.post("/chat")
async def chat(query: str = Form(...)):
    context_docs = retrieve_context(query)
    context = "\n".join([d.page_content for d in context_docs])
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    answer = call_groq_chat(prompt)
    return {"answer": answer}

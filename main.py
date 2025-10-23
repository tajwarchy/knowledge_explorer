# main.py
import os
import time
import uuid
import json
from typing import List

import requests
from dotenv import load_dotenv

# Document handling
import docx2txt
import html2text
from pypdf import PdfReader

# Langchain / embeddings / text splitting / vectorstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Gradio UI
import gradio as gr

# local config
from config import GROQ_API_KEY, CHROMA_PATH, EMBED_MODEL, LLM_MODEL

load_dotenv()


# ----------------------------
# Helpers: document parsers
# ----------------------------
def extract_text_from_pdf(file_path: str) -> str:
    text_chunks = []
    reader = PdfReader(file_path)
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        text_chunks.append(txt)
    return "\n".join(text_chunks)


def extract_text_from_docx(file_path: str) -> str:
    return docx2txt.process(file_path) or ""


def extract_text_from_html(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    return html2text.html2text(html)


def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text_from_upload(uploaded_file) -> str:
    """
    uploaded_file is a temporary file object from Gradio (path attribute available)
    """
    filepath = uploaded_file.name
    _, ext = os.path.splitext(filepath.lower())
    if ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(filepath)
    elif ext in (".html", ".htm"):
        return extract_text_from_html(filepath)
    else:
        # fallback to plain text
        return extract_text_from_txt(filepath)


# ----------------------------
# Embedding + Vectorstore helpers
# ----------------------------
def get_embeddings_instance():

    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def create_or_get_vectorstore(collection_name: str = "knowledge"):
    """
    Create or load a Chroma vectorstore persisted on disk.
    """
    persist_dir = CHROMA_PATH
    os.makedirs(persist_dir, exist_ok=True)

    embeddings = get_embeddings_instance()
    # If a persisted collection exists, LangChain's Chroma wrapper will load it if you pass same persist_directory + collection_name
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="chroma_storage",
    )
    return vectorstore


def ingest_text_to_vectorstore(
    text: str, metadata: dict = None, collection_name: str = "knowledge"
):
    """
    Split text into chunks, create Documents and add to Chroma (persist).
    Uses RecursiveCharacterTextSplitter for balanced chunking.
    """
    if not text or len(text.strip()) == 0:
        return 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )

    # Create a single Document with optional metadata then split
    doc = Document(page_content=text, metadata=metadata or {})
    chunks = text_splitter.split_documents([doc])

    # Create / load vectorstore and add
    vectorstore = create_or_get_vectorstore(collection_name=collection_name)
    # add_documents expects list[Document]
    vectorstore.add_documents(chunks)
    return len(chunks)


# ----------------------------
# Retriever + Prompting
# ----------------------------
def retrieve_context(
    query: str, k: int = 5, collection_name: str = "knowledge"
) -> List[str]:
    """
    Returns top-k relevant chunk texts (page_content).
    """
    vectorstore = create_or_get_vectorstore(collection_name=collection_name)
    # similarity_search returns list of Documents
    results = vectorstore.similarity_search(query, k=k)
    return [d.page_content for d in results]


def build_prompt_with_context(
    context_chunks: List[str], user_question: str
) -> List[dict]:
    """
    Builds a chat-message list for the Groq OpenAI-compatible chat endpoint.
    We include a system instruction and a user message containing the context and actual question.
    """
    # Keep contexts short: join top chunks
    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else ""
    system_msg = (
        "You are a precise, factual AI assistant. Use only the information provided in the context below "
        "to answer the user's question. If the answer is not in the provided context, say you don't know or recommend verifying sources. "
        "Do not hallucinate facts."
    )

    # Construct user content: include context then question
    user_content = f"CONTEXT:\n{context_text}\n\nQUESTION:\n{user_question}\n\nINSTRUCTIONS: Answer concisely and cite the context by quoting short snippets if needed."

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]
    return messages


# ----------------------------
# Groq API call
# ----------------------------
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


def call_groq_chat(
    messages: List[dict],
    model: str = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
):
    """
    Calls Groq's OpenAI-compatible chat completions endpoint.
    Returns raw text reply.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env.")

    payload = {
        "model": model or LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text}")

    data = resp.json()
    # Groq OpenAI-compatible responses usually follow choices -> message -> content
    # but Groq also has other shapes; handle common possibilities.
    try:
        # Standard OpenAI-compatible
        return data["choices"][0]["message"]["content"]
    except Exception:
        # Try alternative top-level output_text (older Groq SDK examples)
        if "output_text" in data:
            return data["output_text"]
        # As a fallback, stringify the JSON
        return json.dumps(data)


# ----------------------------
# Gradio app code
# ----------------------------
def reset_collection(collection_name: str = "knowledge"):
    # Remove persisted folder for the collection to reset state (use with caution)
    # This only removes persisted files; Chroma will recreate on next ingest.
    if os.path.exists(CHROMA_PATH):
        # Be selective: don't delete whole project; remove chroma files only.
        # For simplicity here we'll remove the persist directory entirely.
        import shutil

        shutil.rmtree(CHROMA_PATH)
    os.makedirs(CHROMA_PATH, exist_ok=True)
    # instantiate empty collection
    create_or_get_vectorstore(collection_name=collection_name)


def handle_file_upload(uploaded_file, metadata_name="uploaded"):
    """
    Called when file is uploaded via Gradio.
    Returns (status_message, added_chunks_count)
    """
    if uploaded_file is None:
        return "No file uploaded.", 0

    text = extract_text_from_upload(uploaded_file)
    if not text or len(text.strip()) == 0:
        return "Could not extract text from file.", 0

    meta = {
        "source": getattr(uploaded_file, "name", f"{metadata_name}_{int(time.time())}")
    }
    added = ingest_text_to_vectorstore(
        text=text, metadata=meta, collection_name="knowledge"
    )
    return f"Indexed file: {meta['source']} ({added} chunks added)", added


def chat_with_docs(user_message: str, history, k: int = 5):
    """
    Main chat handler for Gradio: retrieve context, call LLM, append to history.
    history is the Gradio chat history list.
    """
    # Retrieve context chunks
    contexts = retrieve_context(user_message, k=k, collection_name="knowledge")
    messages = build_prompt_with_context(contexts, user_message)
    reply = call_groq_chat(messages=messages)
    # Append to history
    history = history or []
    history.append((user_message, reply))
    return history, history


# ----------------------------
# Run Gradio UI
# ----------------------------
def run_app():
    title = "Knowledge Explorer — Upload & Chat (Groq + Chroma)"
    description = (
        "Upload documents (PDF / DOCX / HTML / TXT) and then ask questions about their content. "
        "Documents are embedded and stored locally. Uses Groq for LLM inference."
    )

    with gr.Blocks() as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)

        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.File(
                    label="Upload document (PDF, DOCX, HTML, TXT)", file_count="single"
                )
                upload_btn = gr.Button("Index file")
                reset_btn = gr.Button(
                    "Reset vector store (clear all indexed documents)"
                )
                status = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat with documents")
                msg = gr.Textbox(
                    label="Your question",
                    placeholder="Ask anything about the uploaded documents...",
                )
                send = gr.Button("Send")
                k_slider = gr.Slider(
                    minimum=1, maximum=8, value=5, step=1, label="Context chunks (k)"
                )

        # Hook up events
        def on_index_click(file):
            return handle_file_upload(file)

        upload_btn.click(fn=on_index_click, inputs=upload, outputs=status)

        def on_reset_click():
            reset_collection()
            return "Vector store reset."

        reset_btn.click(fn=on_reset_click, inputs=None, outputs=status)

        def on_send(user_text, history_list, k):
            history, full_history = chat_with_docs(user_text, history_list, k=int(k))
            # Gradio Chatbot expects list of (user, bot) tuples
            return full_history, ""

        send.click(fn=on_send, inputs=[msg, chatbot, k_slider], outputs=[chatbot, msg])

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    print("Starting Knowledge Explorer...")
    run_app()

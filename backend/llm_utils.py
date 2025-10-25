import requests
from .config import GROQ_API_KEY, LLM_MODEL


def call_groq_chat(prompt: str):
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers based on uploaded documents.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 512,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text}")
    return resp.json()["choices"][0]["message"]["content"]

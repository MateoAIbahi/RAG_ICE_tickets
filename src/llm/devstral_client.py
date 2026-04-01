import os
import requests

DEVSTRAL_BASE_URL = os.getenv("DEVSTRAL_BASE_URL", "http://icechat.ice.local/v1-devstral")
DEVSTRAL_MODEL = os.getenv("DEVSTRAL_MODEL", "ollama/devstral-local")


def ask_devstral(question: str, context: str) -> str:
    url = f"{DEVSTRAL_BASE_URL}/chat/completions"

    payload = {
        "model": DEVSTRAL_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Tu es un assistant technique ICE. "
                    "Tu réponds uniquement à partir du contexte fourni. "
                    "Si l'information n'est pas dans le contexte, dis-le clairement. "
                    "Réponds de manière structurée et précise."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nContexte:\n{context}",
            },
        ],
        "temperature": 0.2,
    }

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"]
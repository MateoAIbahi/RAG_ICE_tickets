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
                    "Tu es un assistant technique ICE.\n"
                    "Tu dois répondre UNIQUEMENT à partir du contexte fourni.\n"
                    "Ne fais aucune supposition.\n"
                    "Si la réponse n'est pas clairement présente dans le contexte, dis : 'Information non trouvée dans les documents fournis'.\n"
                    "Cite les sources utilisées (nom du document et page).\n"
                    "Réponds de manière précise et technique."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Contexte:\n{context}\n\n"
                    "Réponds uniquement à partir du contexte.\n"
                    "Cite les sources sous la forme : (document, page).\n"
                ),
            },
        ],
        "temperature": 0.2,
    }

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"]
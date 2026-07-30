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
                    "Tu réponds UNIQUEMENT à partir du contexte fourni. Ne fais aucune supposition.\n"
                    "Si la réponse n'est pas clairement présente dans le contexte, réponds : "
                    "'Information non trouvée dans les documents fournis'.\n"
                    "\n"
                    "RÈGLES DE RÉPONSE :\n"
                    "- Réponds directement, sans reformuler ni recopier la question.\n"
                    "- Sois précis et technique, sans remplissage.\n"
                    "- Si la question porte sur plusieurs points et que le contexte n'en couvre "
                    "qu'une partie, réponds sur ce qui est couvert et signale explicitement les "
                    "points non trouvés.\n"
                    "- Examine TOUTES les sources fournies avant de répondre, y compris les "
                    "dernières de la liste. La source la plus pertinente n'est pas toujours "
                    "la première.\n"
                    "- Si plusieurs sources apportent des éléments différents, combine-les "
                    "plutôt que de t'arrêter à la première réponse plausible.\n"
                    "- Si le contexte mentionne une condition préalable, un prérequis ou une "
                    "restriction d'usage (mode d'exploitation, état de l'équipement, version "
                    "requise), tu dois l'énoncer explicitement AVANT les étapes de la procédure.\n"
                    "\n"
                    "RÈGLES DE CITATION :\n"
                    "- Cite tes sources en insérant le marqueur [Source N] directement dans le "
                    "texte, juste après l'information concernée.\n"
                    "- N'ajoute AUCUNE liste de sources en fin de réponse : les marqueurs "
                    "[Source N] suffisent, l'affichage est géré en aval.\n"
                    "- N'invente jamais de numéro de page. Les tickets n'ont pas de page.\n"
                    "- Ne cite que les sources que tu as réellement utilisées."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Contexte:\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
        "temperature": 0,
    }

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"]
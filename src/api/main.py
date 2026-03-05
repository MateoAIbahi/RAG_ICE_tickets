from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="RAG ICE Tickets API")

class SearchRequest(BaseModel):
    question: str
    sources: List[str] = ["ticket", "pdf"]  # ticket | pdf | les deux

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/search")
def search(req: SearchRequest):
    # MVP: on ne fait pas encore de DB, juste on renvoie ce qu'on a reçu
    # (on branchera Postgres juste après)
    return {
        "question": req.question,
        "sources_filter": req.sources,
        "results": []
    }
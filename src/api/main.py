import os
import uuid
import json
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
import shutil

from src.query.search import search_similar_documents
from src.llm.devstral_client import ask_devstral

app = FastAPI(title="RAG ICE API")

UPLOAD_DIR = Path("/data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx", ".odt"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 8
    source_type: str | None = None
    pccn_version: str | None = None

class AskRequest(BaseModel):
    query: str
    top_k: int = 8
    source_type: str | None = None
    pccn_version: str | None = None

def get_conn():
    db_url = os.getenv("RAG_DATABASE_URL", "postgresql://rag:ragpass@db:5432/ragdb")
    return psycopg2.connect(db_url)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest):
    query = req.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query vide")

    results = search_similar_documents(
        query=query,
        top_k=req.top_k,
        source_type=req.source_type,
        pccn_version=req.pccn_version,
    )
    return {"results": results}

@app.post("/ask")
def ask(req: AskRequest):
    query = req.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query vide")

    results = search_similar_documents(
        query=query,
        top_k=req.top_k,
        source_type=req.source_type,
        pccn_version=req.pccn_version,
    )

    # 🔥 Construction du contexte
    context_parts = []
    for i, r in enumerate(results, start=1):
        source_label = f"{r['source_type']} | {r['source_path']}"
        if r.get("page_num") is not None:
            source_label += f" | page {r['page_num']}"

        context_parts.append(
            f"[Source {i}] {source_label}\nContenu:\n{r['content']}"
        )

    context = "\n\n".join(context_parts)

    # 🔥 Appel LLM
    answer = ask_devstral(question=query, context=context)

    return {
        "answer": answer,
        "sources": [
            {
                "source_type": r["source_type"],
                "source_path": r["source_path"],
                "page_num": r["page_num"],
            }
            for r in results
        ],
    }
    
@app.post("/upload-pdf")
async def upload_pdf(
    files: List[UploadFile] = File(...),
    pccn_version: str = Form(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    if not pccn_version:
        raise HTTPException(status_code=400, detail="pccn_version missing")

    upload_id = f"upload_{uuid.uuid4().hex}"
    upload_path = UPLOAD_DIR / upload_id
    upload_path.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for file in files:
        suffix = Path(file.filename).suffix.lower()

        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"{file.filename} has unsupported extension"
            )

        file_path = upload_path / file.filename

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        meta_path = upload_path / f"{file.filename}.meta.json"

        with meta_path.open("w") as f:
            json.dump({
                "pccn_version": pccn_version
            }, f)

        saved_files.append(str(file_path))

    return {
        "message": "Upload successful",
        "pccn_version": pccn_version,
        "upload_dir": str(upload_path),
        "files": saved_files
    }
import os
import uuid
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
import shutil
from src.query.search import search_similar_documents

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
    top_k: int = 5
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
    question = req.question.strip()
    sources = req.sources or ["ticket", "pdf"]
    pccn_version = req.pccn_version
    limit = max(1, min(req.limit, 50))

    if not question:
        raise HTTPException(status_code=400, detail="Question vide")

    sql = """
        SELECT
            id,
            source_type,
            source_id,
            source_path,
            page_num,
            row_num,
            chunk_id,
            pccn_version,
            content
        FROM documents
        WHERE source_type = ANY(%s)
    """

    params = [sources]

    if pccn_version:
        sql += " AND pccn_version = %s"
        params.append(pccn_version)

    sql += """
        AND content ILIKE %s
        ORDER BY id DESC
        LIMIT %s
    """
    params.append(f"%{question}%")
    params.append(limit)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    results = []
    for row in rows:
        results.append({
            "id": row[0],
            "source_type": row[1],
            "source_id": row[2],
            "source_path": row[3],
            "page_num": row[4],
            "row_num": row[5],
            "chunk_id": row[6],
            "pccn_version": row[7],
            "content_preview": (row[8] or "")[:500]
        })

    return {
        "question": question,
        "sources_filter": sources,
        "pccn_version_filter": pccn_version,
        "results": results
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

    @app.post("/search")
    def search(req: SearchRequest):
        query = req.query.strip()

        results = search_similar_documents(
            query=query,
            top_k=req.top_k,
            source_type=req.source_type,
            pccn_version=req.pccn_version,
        )
        return {"results": results}
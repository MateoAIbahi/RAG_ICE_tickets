import os
import uuid
import json
from pathlib import Path
from typing import List
import time
import re

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
    top_k: int = 10
    source_type: str | None = None
    pccn_version: str | None = None

class AskRequest(BaseModel):
    query: str
    top_k: int = 10
    source_type: str | None = None
    pccn_version: str | None = None

def get_conn():
    db_url = os.getenv("RAG_DATABASE_URL", "postgresql://rag:ragpass@db:5432/ragdb")
    return psycopg2.connect(db_url)

def build_source_label(r: dict) -> str:
    if r["source_type"] in ("ticket", "mantis"):
        return f"ticket {r['source_path']}"
    label = r["source_path"] or "document"
    if r.get("page_num") is not None:
        label += f", page {r['page_num']}"
    return label

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
    t0 = time.time()

    query = req.query.strip()
    print(f"[ASK] q={query!r} source_type={req.source_type!r} pccn_version={req.pccn_version!r} top_k={req.top_k}")
    print(f"[ASK] query reçue, top_k={req.top_k}")
    if not query:
        raise HTTPException(status_code=400, detail="Query vide")
    results = search_similar_documents(
        query=query,
        top_k=req.top_k,
        source_type=req.source_type,
        pccn_version=req.pccn_version,
    )
    print("[ASK] top:", [
        (r["source_path"], r["page_num"], f"d={r['dense_rank']}", f"l={r['lex_rank']}")
        for r in results[:10]
    ])
    print(f"[ASK] retrieval terminé en {time.time() - t0:.2f}s, nb résultats={len(results)}")
    if not results:
        return {
            "answer": "Aucun document ne correspond aux filtres sélectionnés.",
            "sources_complete": True,
            "sources": [],
        }
    context_parts = []
    for i, r in enumerate(results, start=1):
        context_parts.append(
            f"[Source {i}] {build_source_label(r)}\nContenu:\n{r['content']}"
        )
    context = "\n\n".join(context_parts)

    print(f"[ASK] contexte construit, taille caractères={len(context)}")
    print(f"[ASK] aperçu contexte: {context[:500]}")

    t1 = time.time()
    print("[ASK] appel devstral...")
    answer = ask_devstral(question=query, context=context)
    print(f"[ASK] devstral terminé en {time.time() - t1:.2f}s")
    print(f"[ASK] total /ask = {time.time() - t0:.2f}s")

    cited = {int(n) for n in re.findall(r"\[Source\s+(\d+)\]", answer)}
    cited = {n for n in cited if 1 <= n <= len(results)}

    if cited:
        selected = [(i, results[i - 1]) for i in sorted(cited)]
        sources_complete = True
    else:
        # Le modèle n'a cité aucune source : on renvoie tout plutôt que rien,
        # mais on le signale pour ne pas laisser croire à une attribution fiable.
        selected = list(enumerate(results, start=1))
        sources_complete = False

    print(f"[ASK] sources citées={sorted(cited) or 'aucune'} / {len(results)} remontées")

    return {
        "answer": answer,
        "sources_complete": sources_complete,
        "sources": [
            {
                "index": i,
                "label": build_source_label(r),
                "source_type": r["source_type"],
                "source_path": r["source_path"],
                "page_num": r["page_num"],
            }
            for i, r in selected
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

@app.get("/documents")
def list_documents():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT source_id, source_path, pccn_version,
                       count(*) AS chunks,
                       min(created_at)::date AS ingere_le
                FROM documents
                WHERE source_type = 'pdf' AND deleted_at IS NULL
                GROUP BY source_id, source_path, pccn_version
                ORDER BY source_path
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    return {
        "documents": [
            {"source_id": r[0], "source_path": r[1], "pccn_version": r[2],
             "chunks": r[3], "ingere_le": str(r[4])}
            for r in rows
        ]
    }

class DeleteDocRequest(BaseModel):
    source_id: str
    password: str


@app.post("/documents/delete")
def delete_document(req: DeleteDocRequest):
    expected = os.getenv("ADMIN_PASSWORD", "icetickets")
    if req.password != expected:
        raise HTTPException(status_code=403, detail="Mot de passe incorrect")

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE documents SET deleted_at = now()
                WHERE source_type = 'pdf' AND source_id = %s AND deleted_at IS NULL
            """, (req.source_id,))
            n = cur.rowcount
        conn.commit()
    finally:
        conn.close()

    if n == 0:
        raise HTTPException(status_code=404, detail="Document introuvable")

    print(f"[DOCS] Suppression logique: {req.source_id} ({n} pages)")
    return {"deleted": req.source_id, "pages": n}


@app.get("/documents/deleted")
def list_deleted_documents():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT source_id, source_path, count(*), max(deleted_at)::date
                FROM documents
                WHERE source_type = 'pdf' AND deleted_at IS NOT NULL
                GROUP BY source_id, source_path ORDER BY 4 DESC
            """)
            rows = cur.fetchall()
    finally:
        conn.close()
    return {"documents": [
        {"source_id": r[0], "source_path": r[1], "chunks": r[2], "supprime_le": str(r[3])}
        for r in rows
    ]}
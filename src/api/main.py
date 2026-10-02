import hmac
import os
import re
import time
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil

from src.common.db import get_conn as _get_conn
from src.common.embedder import get_embedder
from src.db.migrate import migrate
from src.ingestion.documents import FAILED_DIRNAME, SUPPORTED_EXT, list_pending, read_meta
from src.query.search import search_similar_documents
from src.llm.devstral_client import ask_devstral

app = FastAPI(title="RAG ICE API")

UPLOAD_DIR = Path(os.getenv("DOCS_INBOX", "/data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = SUPPORTED_EXT
TRANCHES = {"AUTO", "TRA", "RAME"}
PCCN_RE = re.compile(r"^V\d+$")
DICODATA_RE = re.compile(r"^\d+(\.\d+)*$")


# --- Mode maintenance -------------------------------------------------------
# Activé/désactivé par deploy/maintenance.sh (présence d'un fichier drapeau).
# Pendant la maintenance, toutes les routes sauf /status et /health répondent 503
# et l'interface affiche un écran "Service en maintenance".
MAINTENANCE_FLAG = Path(os.getenv("MAINTENANCE_FLAG", "/data/state/maintenance"))
DEFAULT_MAINTENANCE_MESSAGE = (
    "L'assistant est en cours de mise à jour : les documents et les tickets sont "
    "en train d'être réintégrés. Il sera de nouveau disponible à la fin de l'opération."
)
ALWAYS_OPEN = {"/status", "/health"}


def maintenance_message() -> str | None:
    try:
        if MAINTENANCE_FLAG.exists():
            return MAINTENANCE_FLAG.read_text(encoding="utf-8").strip() or DEFAULT_MAINTENANCE_MESSAGE
    except OSError:
        return DEFAULT_MAINTENANCE_MESSAGE
    return None


@app.middleware("http")
async def maintenance_gate(request: Request, call_next):
    if request.url.path not in ALWAYS_OPEN:
        msg = maintenance_message()
        if msg:
            return JSONResponse(status_code=503, content={"detail": msg, "maintenance": True})
    return await call_next(request)


def reindex_progress() -> dict | None:
    """Avancement indicatif : part des éléments actifs qui ont un embedding."""
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""SELECT count(*), count(embedding)
                                 FROM documents WHERE deleted_at IS NULL""")
                total, done = cur.fetchone()
        finally:
            conn.close()
        return {"total": total, "done": done,
                "percent": round(100 * done / total) if total else None}
    except Exception:
        return None


@app.get("/status")
def status():
    msg = maintenance_message()
    return {
        "maintenance": msg is not None,
        "message": msg,
        "progress": reindex_progress() if msg else None,
    }


@app.on_event("startup")
def _startup():
    migrate()
    get_embedder()  # chargé une fois au démarrage plutôt qu'à la 1re question


def check_password(password: str | None):
    """Vérification côté serveur (avant : uniquement dans le JavaScript pour l'ajout)."""
    expected = os.getenv("ADMIN_PASSWORD")
    if not expected:
        raise HTTPException(status_code=503, detail="ADMIN_PASSWORD non configuré sur le serveur")
    if not password or not hmac.compare_digest(password.encode(), expected.encode()):
        raise HTTPException(status_code=403, detail="Mot de passe incorrect")

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
    tranche: str | None = None

class AskRequest(BaseModel):
    query: str
    top_k: int = 10
    source_type: str | None = None
    pccn_version: str | None = None
    tranche: str | None = None

def get_conn():
    return _get_conn(autocommit=False)

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
        tranche=req.tranche,
    )
    return {"results": results}

@app.post("/ask")
def ask(req: AskRequest):
    t0 = time.time()

    query = req.query.strip()
    print(f"[ASK] q={query!r} source_type={req.source_type!r} pccn_version={req.pccn_version!r} tranche={req.tranche!r} top_k={req.top_k}")
    print(f"[ASK] query reçue, top_k={req.top_k}")
    if not query:
        raise HTTPException(status_code=400, detail="Query vide")
    results = search_similar_documents(
        query=query,
        top_k=req.top_k,
        source_type=req.source_type,
        pccn_version=req.pccn_version,
        tranche=req.tranche,
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
                "pccn_version": r.get("pccn_version"),
                "tranche": r.get("tranche"),
                "dicodata_version": r.get("dicodata_version"),
            }
            for i, r in selected
        ],
    }
    
class PasswordRequest(BaseModel):
    password: str


@app.post("/auth/check")
def auth_check(req: PasswordRequest):
    check_password(req.password)
    return {"ok": True}


def _safe_filename(name: str | None) -> str:
    # Path(...).name retire tout chemin (../../etc) envoyé par le navigateur
    clean = Path((name or "").replace("\\", "/")).name.strip()
    if not clean or clean in {".", ".."}:
        raise HTTPException(status_code=400, detail="Nom de fichier invalide")
    return clean


@app.post("/upload-pdf")
async def upload_pdf(
    files: List[UploadFile] = File(...),
    pccn_version: str = Form(...),
    password: str = Form(None),
    tranche: str = Form(""),
    dicodata_version: str = Form(""),
):
    check_password(password)

    pccn_version = (pccn_version or "").strip().upper()
    tranche = (tranche or "").strip().upper() or None
    dicodata_version = (dicodata_version or "").strip().replace(",", ".") or None

    if not PCCN_RE.match(pccn_version):
        raise HTTPException(status_code=400, detail="Version PCCN invalide (attendu V1, V2...)")
    if tranche and tranche not in TRANCHES:
        raise HTTPException(status_code=400, detail=f"Tranche invalide : {tranche}")
    if dicodata_version and not DICODATA_RE.match(dicodata_version):
        raise HTTPException(status_code=400, detail="Version Dicodata invalide (ex. 3.2)")
    if dicodata_version:
        tranche = None  # Dicodata : communes à toutes les tranches du palier
    if not files:
        raise HTTPException(status_code=400, detail="Aucun fichier")

    names = [_safe_filename(f.filename) for f in files]
    for n in names:
        if Path(n).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"{n} : format non géré")

    upload_path = UPLOAD_DIR / f"upload_{uuid.uuid4().hex}"
    upload_path.mkdir(parents=True, exist_ok=True)

    saved = []
    for file, name in zip(files, names):
        dest = upload_path / name
        with dest.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        if dest.stat().st_size == 0:
            dest.unlink()
            raise HTTPException(status_code=400, detail=f"{name} est vide")
        meta = {
            "pccn_version": pccn_version,
            "tranche": tranche,
            "dicodata_version": dicodata_version,
            "relpath": name,
            "origin": "upload",
            "uploaded_at": datetime.now().isoformat(timespec="seconds"),
        }
        (upload_path / f"{name}.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
        saved.append(name)

    return {
        "message": "Fichiers reçus. Ils seront intégrés lors de la prochaine ingestion (vendredi soir).",
        "pccn_version": pccn_version,
        "tranche": tranche,
        "dicodata_version": dicodata_version,
        "files": saved,
    }


@app.get("/documents")
def list_documents():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT source_id,
                       min(source_path),
                       min(pccn_version),
                       min(tranche),
                       min(dicodata_version),
                       count(DISTINCT page_num) AS pages,
                       count(*) AS chunks,
                       min(created_at)::date,
                       bool_or(metadata->>'legacy' = 'true') AS legacy
                FROM documents
                WHERE source_type = 'pdf' AND deleted_at IS NULL
                GROUP BY source_id
                ORDER BY 3, 2
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    return {"documents": [
        {"source_id": r[0], "source_path": r[1], "pccn_version": r[2], "tranche": r[3],
         "dicodata_version": r[4], "pages": r[5], "chunks": r[6], "ingere_le": str(r[7]),
         "legacy": bool(r[8])}
        for r in rows
    ]}


@app.get("/documents/pending")
def list_pending_documents():
    """Fichiers reçus mais pas encore ingérés, et échecs d'ingestion."""
    pending = []
    for p in list_pending():
        meta = read_meta(p) or {}
        pending.append({
            "file": meta.get("relpath") or p.name,
            "pccn_version": meta.get("pccn_version"),
            "tranche": meta.get("tranche"),
            "dicodata_version": meta.get("dicodata_version"),
            "recu_le": meta.get("uploaded_at") or meta.get("imported_at"),
        })
    failed = []
    failed_root = UPLOAD_DIR / FAILED_DIRNAME
    if failed_root.exists():
        for err in sorted(failed_root.rglob("*.erreur.txt")):
            failed.append({
                "file": err.name[: -len(".erreur.txt")],
                "raison": err.read_text(encoding="utf-8")[:300],
            })
    return {"pending": pending, "failed": failed}


class DocActionRequest(BaseModel):
    source_id: str
    password: str


@app.post("/documents/delete")
def delete_document(req: DocActionRequest):
    check_password(req.password)
    conn = get_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE documents SET deleted_at = now()
                WHERE source_type = 'pdf' AND source_id = %s AND deleted_at IS NULL
            """, (req.source_id,))
            n = cur.rowcount
    finally:
        conn.close()

    if n == 0:
        raise HTTPException(status_code=404, detail="Document introuvable")
    print(f"[DOCS] Suppression logique: {req.source_id} ({n} chunks)")
    return {"deleted": req.source_id, "chunks": n}


@app.post("/documents/restore")
def restore_document(req: DocActionRequest):
    check_password(req.password)
    conn = get_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE documents SET deleted_at = NULL
                WHERE source_type = 'pdf' AND source_id = %s AND deleted_at IS NOT NULL
            """, (req.source_id,))
            n = cur.rowcount
    finally:
        conn.close()

    if n == 0:
        raise HTTPException(status_code=404, detail="Document introuvable dans la corbeille")
    print(f"[DOCS] Restauration: {req.source_id} ({n} chunks)")
    return {"restored": req.source_id, "chunks": n}


@app.get("/documents/deleted")
def list_deleted_documents():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT source_id, min(source_path), min(pccn_version),
                       count(DISTINCT page_num), max(deleted_at)::date
                FROM documents
                WHERE source_type = 'pdf' AND deleted_at IS NOT NULL
                  AND NOT (metadata ? 'replaced_by')   -- remplacé par une réingestion
                GROUP BY source_id ORDER BY 5 DESC
            """)
            rows = cur.fetchall()
    finally:
        conn.close()
    return {"documents": [
        {"source_id": r[0], "source_path": r[1], "pccn_version": r[2],
         "pages": r[3], "supprime_le": str(r[4])}
        for r in rows
    ]}

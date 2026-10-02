"""
Ingestion des documents déposés dans la boîte d'entrée (DOCS_INBOX).

Chaque fichier est accompagné de `<nom>.meta.json` :
    {"pccn_version": "V1", "tranche": "AUTO" | null, "dicodata_version": "3.2" | null,
     "relpath": "PCCN V1/AUTO/xxx.pdf", "origin": "upload" | "import"}

Changements par rapport à l'ancienne version :
- identifiant de document = hash(version, tranche, dicodata, chemin) et non plus
  le nom de fichier seul (deux "Procédure.pdf" en V1 et V2 s'écrasaient) ;
- texte extrait + OCR si la page n'a pas de texte (pages scannées) ;
- découpage en chunks qui tiennent dans le modèle (plus de troncature) ;
- remplacement atomique de toutes les pages d'un document ;
- le fichier d'origine est ARCHIVÉ (DOCS_ARCHIVE) au lieu d'être supprimé,
  ce qui permet une réindexation future ;
- les échecs définitifs (format non géré, aucun texte) partent dans _echecs/
  avec la raison, au lieu de boucler chaque semaine.
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import fitz  # PyMuPDF

from src.common.db import get_conn
from src.common.embedder import get_embedder
from src.ingestion.store import replace_source

INBOX = Path(os.getenv("DOCS_INBOX", "/data/uploads"))
ARCHIVE = Path(os.getenv("DOCS_ARCHIVE", "/data/archive"))
FAILED_DIRNAME = "_echecs"

PDF_EXT = {".pdf"}
CONVERTIBLE_EXT = {".doc", ".docx", ".odt", ".rtf", ".ppt", ".pptx", ".odp"}
SUPPORTED_EXT = PDF_EXT | CONVERTIBLE_EXT
IGNORED_NAMES = {"thumbs.db", ".ds_store", "desktop.ini"}

OCR_ENABLED = os.getenv("OCR_ENABLED", "1") == "1"
OCR_MIN_CHARS = int(os.getenv("OCR_MIN_CHARS", "40"))
OCR_LANG = os.getenv("OCR_LANG", "fra+eng")


class PermanentError(Exception):
    """Erreur qui ne se corrigera pas en réessayant (fichier à déplacer en _echecs)."""


def doc_uid(meta: dict) -> str:
    key = "|".join([
        meta.get("pccn_version") or "",
        meta.get("tranche") or "",
        meta.get("dicodata_version") or "",
        (meta.get("relpath") or "").replace("\\", "/").lower(),
    ])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]


def doc_header(meta: dict) -> str:
    parts = [f"Document : {meta.get('relpath')}"]
    if meta.get("pccn_version"):
        parts.append(f"PCCN {meta['pccn_version']}")
    parts.append(f"Tranche {meta['tranche']}" if meta.get("tranche") else "Toutes tranches")
    if meta.get("dicodata_version"):
        parts.append(f"Dicodata {meta['dicodata_version']}")
    return " | ".join(parts)


def read_meta(path: Path) -> dict | None:
    meta_file = path.parent / f"{path.name}.meta.json"
    if not meta_file.exists():
        return None
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta.setdefault("relpath", path.name)  # anciens uploads : seul pccn_version
    meta.setdefault("tranche", None)
    meta.setdefault("dicodata_version", None)
    return meta


def list_pending() -> list[Path]:
    if not INBOX.exists():
        return []
    return sorted(
        p for p in INBOX.rglob("*")
        if p.is_file()
        and not p.name.endswith(".meta.json")
        and not p.name.endswith(".erreur.txt")
        and FAILED_DIRNAME not in p.relative_to(INBOX).parts
        and p.name.lower() not in IGNORED_NAMES
        and not p.name.startswith("~$")
    )


def convert_to_pdf(src: Path, outdir: Path) -> Path:
    cmd = [
        "soffice", "--headless", "--norestore",
        f"-env:UserInstallation=file://{outdir}/lo_profile",
        "--convert-to", "pdf", "--outdir", str(outdir), str(src),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    pdf = outdir / f"{src.stem}.pdf"
    if res.returncode != 0 or not pdf.exists():
        raise PermanentError(f"Conversion PDF impossible : {res.stderr.strip()[:500]}")
    return pdf


_ocr_warned = False


def extract_pages(pdf: Path) -> list[tuple[int, str, bool]]:
    """[(numéro_page, texte, ocr_utilisé)]"""
    global _ocr_warned
    pages = []
    try:
        doc = fitz.open(pdf)
    except Exception as e:
        raise PermanentError(f"PDF illisible : {e}")
    with doc:
        if doc.needs_pass:
            raise PermanentError("PDF protégé par mot de passe")
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            used_ocr = False
            if OCR_ENABLED and len(text) < OCR_MIN_CHARS:
                try:
                    tp = page.get_textpage_ocr(language=OCR_LANG, dpi=200, full=True)
                    ocr_text = page.get_text("text", textpage=tp).strip()
                    if len(ocr_text) > len(text):
                        text, used_ocr = ocr_text, True
                except Exception as e:
                    if not _ocr_warned:
                        print(f"[DOCS] OCR indisponible ({e}) : pages scannées ignorées")
                        _ocr_warned = True
            pages.append((i, text, used_ocr))
    return pages


def ocr_image_b64(image_b64: str) -> str:
    """
    OCR d'une image PNG en base64 (anciennes pages stockées par la v1, en basse
    résolution). L'image est posée sur une page puis rendue à 300 dpi pour
    l'agrandir avant tesseract. Renvoie "" si l'OCR est indisponible ou vide.
    """
    import base64

    try:
        pix = fitz.Pixmap(base64.b64decode(image_b64))
        doc = fitz.open()
        page = doc.new_page(width=pix.width, height=pix.height)
        page.insert_image(page.rect, pixmap=pix)
        tp = page.get_textpage_ocr(language=OCR_LANG, dpi=300, full=True)
        text = page.get_text("text", textpage=tp).strip()
        doc.close()
        return text
    except Exception as e:
        print(f"[DOCS] OCR image legacy impossible : {e}")
        return ""


def legacy_cleanup(conn, meta: dict, pdf_name: str, uid: str):
    """Retire l'ancienne version (identifiant = nom de fichier) du même document."""
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE documents
                      SET deleted_at = now(),
                          metadata = metadata || jsonb_build_object('replaced_by', %s::text)
                    WHERE source_type = 'pdf'
                      AND metadata->>'legacy' = 'true'
                      AND deleted_at IS NULL
                      AND source_path = %s
                      AND pccn_version IS NOT DISTINCT FROM %s""",
                (uid, pdf_name, meta.get("pccn_version")),
            )
            return cur.rowcount


def ingest_file(conn, path: Path, meta: dict) -> dict:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise PermanentError(f"Format non géré : {ext or '(sans extension)'}")

    embedder = get_embedder()
    uid = doc_uid(meta)
    header = doc_header(meta)

    with tempfile.TemporaryDirectory() as tmp:
        pdf = path if ext in PDF_EXT else convert_to_pdf(path, Path(tmp))
        pages = extract_pages(pdf)

    chunks = []
    ocr_pages = empty_pages = 0
    for page_num, text, used_ocr in pages:
        ocr_pages += used_ocr
        pieces = embedder.chunk(text, header)
        if not pieces:
            empty_pages += 1
        for j, piece in enumerate(pieces):
            chunks.append({
                "source_path": meta["relpath"],
                "page_num": page_num,
                "chunk_id": f"p{page_num}-c{j}",
                "content": f"{header} | page {page_num}\n{piece}",
                "metadata": {
                    "doc_uid": uid,
                    "file_name": path.name,
                    "origin": meta.get("origin"),
                    "ocr": used_ocr,
                },
                "pccn_version": meta.get("pccn_version"),
                "tranche": meta.get("tranche"),
                "dicodata_version": meta.get("dicodata_version"),
            })

    if not chunks:
        raise PermanentError(f"Aucun texte exploitable ({len(pages)} page(s), même après OCR)")

    vectors = embedder.embed_documents([c["content"] for c in chunks])
    for c, v in zip(chunks, vectors):
        c["embedding"] = v

    replace_source(conn, "pdf", uid, chunks)
    removed_legacy = legacy_cleanup(conn, meta, Path(path.name).with_suffix(".pdf").name, uid)

    return {
        "uid": uid, "pages": len(pages), "chunks": len(chunks),
        "ocr_pages": ocr_pages, "empty_pages": empty_pages, "legacy_removed": removed_legacy,
    }


def _move_with_meta(path: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    meta_file = path.parent / f"{path.name}.meta.json"
    shutil.move(str(path), str(dest_dir / path.name))
    if meta_file.exists():
        shutil.move(str(meta_file), str(dest_dir / meta_file.name))


def _cleanup_empty_dirs():
    for d in sorted((p for p in INBOX.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
        if d.name == FAILED_DIRNAME:
            continue
        try:
            d.rmdir()
        except OSError:
            pass


def ingest_documents() -> dict:
    files = list_pending()
    print(f"[DOCS] {len(files)} fichier(s) en attente dans {INBOX}")
    stats = {"ok": 0, "echecs": 0, "a_reessayer": 0, "sans_meta": 0}
    if not files:
        return stats

    conn = get_conn(autocommit=False)
    try:
        for path in files:
            meta = read_meta(path)
            if meta is None:
                print(f"[DOCS] IGNORÉ (pas de .meta.json) : {path}")
                stats["sans_meta"] += 1
                continue
            try:
                info = ingest_file(conn, path, meta)
                _move_with_meta(path, ARCHIVE / info["uid"])
                stats["ok"] += 1
                print(f"[DOCS] OK {meta['relpath']} -> {info}")
            except PermanentError as e:
                failed_dir = INBOX / FAILED_DIRNAME / path.parent.relative_to(INBOX)
                _move_with_meta(path, failed_dir)
                (failed_dir / f"{path.name}.erreur.txt").write_text(str(e), encoding="utf-8")
                stats["echecs"] += 1
                print(f"[DOCS] ÉCHEC {meta['relpath']} : {e}")
            except Exception as e:
                # Erreur transitoire (DB, mémoire...) : le fichier reste, retenté la semaine suivante
                conn.rollback()
                stats["a_reessayer"] += 1
                print(f"[DOCS] ERREUR (retentée au prochain passage) {meta['relpath']} : {e!r}")
    finally:
        conn.close()
        _cleanup_empty_dirs()

    print(f"[DOCS] Bilan : {stats}")
    return stats


if __name__ == "__main__":
    ingest_documents()

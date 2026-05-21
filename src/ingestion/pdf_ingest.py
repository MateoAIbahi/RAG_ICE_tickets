import os
import json
import base64
import io
from pathlib import Path

import fitz  # PyMuPDF
import psycopg2
import torch
from PIL import Image

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

def clean_text_for_postgres(text: str) -> str:
    if text is None:
        return ""
    return text.replace("\x00", "")

def get_rag_conn():
    db_url = os.getenv("RAG_DATABASE_URL", "postgresql://rag:ragpass@db:5432/ragdb")
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    return conn


def read_pdf_metadata(pdf_path: Path):
    meta_file = pdf_path.parent / f"{pdf_path.name}.meta.json"

    if not meta_file.exists():
        print(f"[PDF] Metadata not found for {pdf_path}")
        return None

    try:
        with meta_file.open() as f:
            data = json.load(f)
        return data.get("pccn_version")
    except Exception as e:
        print(f"[PDF] Error reading metadata {meta_file}: {e}")
        return None


def load_model():
    model_name = os.getenv("EMBED_MODEL", "Metric-AI/ColQwen2.5-3b-multilingual-v1.0")
    device = os.getenv("DEVICE", "cpu")

    print(f"[PDF] Loading model: {model_name}")
    print(f"[PDF] Device: {device}")

    model = ColQwen2_5.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
    ).eval()

    processor = ColQwen2_5_Processor.from_pretrained(model_name)
    return model, processor, device


def pil_image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def iter_pdf_pages(pdf_path: Path, zoom: float = 1.5):
    """
    Génère les pages une par une pour éviter de charger tout le PDF en RAM.
    Retourne à chaque itération :
    (page_num, image_pil, extracted_text, page_image_base64)
    """
    doc = fitz.open(pdf_path)

    try:
        for i, page in enumerate(doc):
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, alpha=False)

            img = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            text = page.get_text("text").strip()
            image_b64 = pil_image_to_base64(img)

            yield i + 1, img, text, image_b64

    finally:
        doc.close()

def pool_embedding(embeddings: torch.Tensor):
    """
    embeddings attendu :
    - soit (1, n_vectors, 128)
    - soit (n_vectors, 128)
    Retourne une liste Python de taille 128
    """
    if embeddings.dim() == 3:
        pooled = embeddings.mean(dim=1).squeeze(0)
    elif embeddings.dim() == 2:
        pooled = embeddings.mean(dim=0)
    else:
        raise ValueError(f"Shape inattendue pour embeddings: {tuple(embeddings.shape)}")

    return pooled.detach().cpu().tolist()


def embed_image(image: Image.Image, model, processor, device: str):
    batch = processor.process_images([image]).to(device)

    with torch.no_grad():
        embeddings = model(**batch)

    return pool_embedding(embeddings)


def insert_document(
    conn,
    source_id: str,
    source_path: str,
    page_num: int,
    chunk_id: str,
    content: str,
    pooled_embedding: list,
    pccn_version: str,
    page_image_base64: str,
):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (
                source_type,
                source_id,
                source_path,
                page_num,
                chunk_id,
                content,
                pooled_embedding,
                pccn_version,
                page_image_base64
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s)
            ON CONFLICT (source_type, source_id, chunk_id)
            DO UPDATE SET
                source_path = EXCLUDED.source_path,
                page_num = EXCLUDED.page_num,
                content = EXCLUDED.content,
                pooled_embedding = EXCLUDED.pooled_embedding,
                pccn_version = EXCLUDED.pccn_version,
                page_image_base64 = EXCLUDED.page_image_base64
            """,
            (
                "pdf",
                source_id,
                source_path,
                page_num,
                chunk_id,
                content,
                str(pooled_embedding),
                pccn_version,
                page_image_base64,
            ),
        )


def ingest_pdf_file(pdf_path: str):
    """
    Ingestion d'un PDF :
    - découpage par page
    - rendu de chaque page en image
    - base64 de l'image
    - embedding image via ColQwen
    - pooling en vecteur 128
    - insertion dans documents
    - suppression du PDF si tout est OK
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF introuvable: {pdf_path}")

    source_id = pdf_path.stem
    source_path = pdf_path.name
    pccn_version = read_pdf_metadata(pdf_path)

    model, processor, device = load_model()
    conn = get_rag_conn()
    page_count = 0

    try:
        for page_num, image, extracted_text, image_b64 in iter_pdf_pages(pdf_path):
            page_count += 1
            chunk_id = f"{source_id}#p{page_num}"

            pooled_embedding = embed_image(image, model, processor, device)

            content = clean_text_for_postgres(extracted_text) if extracted_text else f"[PAGE_IMAGE_ONLY] {source_id} page {page_num}"

            insert_document(
                conn=conn,
                source_id=source_id,
                source_path=source_path,
                page_num=page_num,
                chunk_id=chunk_id,
                content=content,
                pooled_embedding=pooled_embedding,
                pccn_version=pccn_version,
                page_image_base64=image_b64,
            )

            print(f"[PDF] Insert OK: {chunk_id}")
        if page_count == 0:
            print(f"[PDF] Aucun contenu trouvé dans {pdf_path}")
            return
        pdf_path.unlink(missing_ok=True)
        print(f"[PDF] PDF supprimé après ingestion: {pdf_path}")

        meta_file = pdf_path.parent / f"{pdf_path.name}.meta.json"
        meta_file.unlink(missing_ok=True)

        try:
            pdf_path.parent.rmdir()
        except OSError:
            pass

    finally:
        conn.close()
import os
from pathlib import Path
import json
import fitz  # PyMuPDF
import psycopg2
import torch
from PIL import Image

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor


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


def render_pdf_pages(pdf_path: Path, zoom: float = 1.5):
    """
    Rend chaque page du PDF en image PIL + extrait le texte brut de la page.
    Retourne une liste de tuples : (page_num, image_pil, extracted_text)
    """
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        # rendu image
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        img = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        # texte brut éventuel
        text = page.get_text("text").strip()

        pages.append((i + 1, img, text))

    doc.close()
    return pages


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
                pccn_version
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s::vector,%s)
            ON CONFLICT (source_type, source_id, chunk_id)
            DO UPDATE SET
                source_path = EXCLUDED.source_path,
                page_num = EXCLUDED.page_num,
                content = EXCLUDED.content,
                pooled_embedding = EXCLUDED.pooled_embedding,
                pccn_version = EXCLUDED.pccn_version
            """,
            (
                "pdf",
                source_id,
                source_path,
                page_num,
                chunk_id,
                content,
                str(pooled_embedding),
                pccn_version
            ),
        )


def ingest_pdf_file(pdf_path: str):
    """
    Ingestion d'un PDF :
    - découpage par page
    - rendu de chaque page en image
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
    pages = render_pdf_pages(pdf_path)

    if not pages:
        print(f"[PDF] Aucun contenu trouvé dans {pdf_path}")
        return

    model, processor, device = load_model()
    conn = get_rag_conn()

    try:
        for page_num, image, extracted_text in pages:
            chunk_id = f"{source_id}#p{page_num}"

            # Embedding basé sur l'image de la page
            pooled_embedding = embed_image(image, model, processor, device)

            # On garde le texte extrait si dispo, sinon un placeholder
            content = extracted_text if extracted_text else f"[PAGE_IMAGE_ONLY] {source_id} page {page_num}"

            insert_document(
                conn=conn,
                source_id=source_id,
                source_path=source_path,
                page_num=page_num,
                chunk_id=chunk_id,
                content=content,
                pooled_embedding=pooled_embedding,
                pccn_version=pccn_version,
            )

            print(f"[PDF] Insert OK: {chunk_id}")

        # suppression seulement si tout s'est bien passé
        pdf_path.unlink(missing_ok=True)
        print(f"[PDF] PDF supprimé après ingestion: {pdf_path}")

        # on supprime le dossier parent s'il est vide
        try:
            pdf_path.parent.rmdir()
        except OSError:
            pass

    finally:
        conn.close()
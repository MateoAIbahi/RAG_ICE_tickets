import os
import psycopg2
import torch

# Patch compat PEFT / Transformers pour qwen2_vl
try:
    from transformers.integrations import peft as hf_peft_integration

    moe_map = getattr(hf_peft_integration, "_MOE_TARGET_MODULE_MAPPING", None)
    if isinstance(moe_map, dict) and "qwen2_vl" not in moe_map:
        if "qwen2_5_vl" in moe_map:
            moe_map["qwen2_vl"] = moe_map["qwen2_5_vl"]
        else:
            moe_map["qwen2_vl"] = []
except Exception:
    pass

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor


MODEL = None
PROCESSOR = None
DEVICE = None


def get_rag_conn():
    db_url = os.getenv("RAG_DATABASE_URL", "postgresql://rag:ragpass@db:5432/ragdb")
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    return conn


def load_model():
    model_name = os.getenv("EMBED_MODEL", "Metric-AI/ColQwen2.5-3b-multilingual-v1.0")
    device = os.getenv("DEVICE", "cpu")

    print(f"[QUERY] Loading model: {model_name}")
    print(f"[QUERY] Device: {device}")

    model = ColQwen2_5.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
    ).eval()

    processor = ColQwen2_5_Processor.from_pretrained(model_name)
    return model, processor, device


def get_model():
    global MODEL, PROCESSOR, DEVICE

    if MODEL is None or PROCESSOR is None or DEVICE is None:
        MODEL, PROCESSOR, DEVICE = load_model()

    return MODEL, PROCESSOR, DEVICE


def pool_embedding(embeddings: torch.Tensor):
    if embeddings.dim() == 3:
        pooled = embeddings.mean(dim=1).squeeze(0)
    elif embeddings.dim() == 2:
        pooled = embeddings.mean(dim=0)
    else:
        raise ValueError(f"Unexpected embedding shape: {tuple(embeddings.shape)}")

    return pooled.detach().cpu().tolist()


def embed_query(query: str, model, processor, device: str):
    batch = processor.process_queries([query]).to(device)

    with torch.no_grad():
        embeddings = model(**batch)

    return pool_embedding(embeddings)


def search_similar_documents(
    query: str,
    top_k: int = 5,
    source_type: str | None = None,
    pccn_version: str | None = None,
):
    model, processor, device = get_model()
    query_embedding = embed_query(query, model, processor, device)

    conn = get_rag_conn()
    try:
        sql = """
            SELECT
                id,
                source_type,
                source_id,
                source_path,
                page_num,
                chunk_id,
                content,
                metadata,
                pooled_embedding <=> %s::vector AS distance
            FROM documents
            WHERE pooled_embedding IS NOT NULL
        """
        params = [str(query_embedding)]

        if source_type and source_type != "all":
            sql += " AND source_type = %s"
            params.append(source_type)

        if pccn_version:
            sql += " AND pccn_version = %s"
            params.append(pccn_version)

        sql += """
            ORDER BY pooled_embedding <=> %s::vector
            LIMIT %s
        """
        params.append(str(query_embedding))
        params.append(top_k)

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
                "chunk_id": row[5],
                "content": row[6],
                "metadata": row[7],
                "distance": float(row[8]),
            })

        return results
    finally:
        conn.close()
import os
import psycopg2
import torch

from colpali_engine.models.qwen2_5 import ColQwen2_5, ColQwen2_5_Processor
import re

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def extract_technical_tokens(query: str) -> list[str]:
    """Sigles, codes et références : ce que la recherche vectorielle rate."""
    tokens = []
    for tok in TOKEN_RE.findall(query):
        has_digit = any(c.isdigit() for c in tok)
        has_alpha = any(c.isalpha() for c in tok)
        if "_" in tok:
            tokens.append(tok)                      # DCLT_INTER_EQ_GRP
        elif has_digit and has_alpha:
            tokens.append(tok)                      # TR313, 3I0, TES606, NICER2
        elif has_digit and len(tok) >= 3:
            tokens.append(tok)                      # 1188, 2555
        elif has_alpha and tok.isupper() and len(tok) >= 2:
            tokens.append(tok)                      # APT, SAA, TMS, HTB
    return tokens

def select_discriminant_tokens(conn, tokens, filters, params, max_ratio=0.05):
    """Ne garde que les tokens rares : au-delà de 5% du corpus, c'est du bruit."""
    if not tokens:
        return []
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT count(*) FROM documents WHERE pooled_embedding IS NOT NULL {filters}",
            params,
        )
        total = cur.fetchone()[0]
        if not total:
            return []
        kept = []
        for tok in tokens:
            p = dict(params, tok=tok)
            cur.execute(
                f"""SELECT count(*) FROM documents
                    WHERE pooled_embedding IS NOT NULL {filters}
                      AND content_tsv @@ plainto_tsquery('french', %(tok)s)""",
                p,
            )
            n = cur.fetchone()[0]
            if 0 < n <= max_ratio * total:
                kept.append(tok)
            else:
                print(f"[SEARCH] token écarté: {tok} ({n}/{total})")
        return kept


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
    candidates: int = 50,
    rrf_k: int = 60,
    guaranteed_dense: int = 6,
):
    model, processor, device = get_model()
    query_embedding = embed_query(query, model, processor, device)

    filters = ""
    filter_params = {}
    if source_type and source_type != "all":
        filters += " AND source_type = %(source_type)s"
        filter_params["source_type"] = source_type

    if pccn_version and pccn_version != "all":
        filters += " AND (pccn_version = %(pccn_version)s OR pccn_version IS NULL)"
        filter_params["pccn_version"] = pccn_version

    conn = get_rag_conn()
    try:
        tech = select_discriminant_tokens(
            conn, extract_technical_tokens(query), filters, filter_params
        )
        print(f"[SEARCH] tokens retenus={tech or 'aucun'}")

        params = dict(filter_params)
        params.update({
            "qv": str(query_embedding),
            "qlex": " ".join(tech) if tech else query,
            "use_or": bool(tech),
            "cand": candidates,
            "rrf_k": rrf_k,
            "top_k": top_k,
            "fetch": top_k * 3,
        })

        sql = f"""
            WITH dense AS (
                SELECT
                    id,
                    pooled_embedding <=> %(qv)s::vector AS distance,
                    row_number() OVER (ORDER BY pooled_embedding <=> %(qv)s::vector) AS rnk
                FROM documents
                WHERE pooled_embedding IS NOT NULL
                  {filters}
                ORDER BY pooled_embedding <=> %(qv)s::vector
                LIMIT %(cand)s
            ),
            lex AS (
                SELECT
                    id,
                    row_number() OVER (ORDER BY ts_rank_cd(content_tsv, q.tq) DESC) AS rnk
                FROM documents,
                     (SELECT CASE WHEN %(use_or)s
                                  THEN replace(websearch_to_tsquery('french', %(qlex)s)::text,
                                               '&', '|')::tsquery
                                  ELSE websearch_to_tsquery('french', %(qlex)s)
                             END AS tq) q
                WHERE content_tsv @@ q.tq
                  {filters}
                ORDER BY ts_rank_cd(content_tsv, q.tq) DESC
                LIMIT %(cand)s
            ),
            fused AS (
                SELECT
                    COALESCE(d.id, l.id) AS id,
                    COALESCE(1.0 / (%(rrf_k)s + d.rnk), 0)
                      + COALESCE(1.0 / (%(rrf_k)s + l.rnk), 0) AS rrf_score,
                    d.rnk AS dense_rank,
                    l.rnk AS lex_rank,
                    d.distance AS distance
                FROM dense d
                FULL OUTER JOIN lex l ON d.id = l.id
            )
            SELECT
                doc.id,
                doc.source_type,
                doc.source_id,
                doc.source_path,
                doc.page_num,
                doc.chunk_id,
                doc.content,
                doc.metadata,
                doc.pccn_version,
                f.rrf_score,
                f.dense_rank,
                f.lex_rank,
                f.distance
            FROM fused f
            JOIN documents doc ON doc.id = f.id
            ORDER BY f.rrf_score DESC, f.distance ASC NULLS LAST
            LIMIT %(fetch)s
        """

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
                "pccn_version": row[8],
                "rrf_score": float(row[9]),
                "dense_rank": row[10],
                "lex_rank": row[11],
                "distance": float(row[12]) if row[12] is not None else None,
            })

        reserved = [
            r for r in results
            if r["dense_rank"] is not None and r["dense_rank"] <= guaranteed_dense
        ]
        selected = list(reserved)
        seen = {r["id"] for r in selected}
        for r in results:
            if len(selected) >= top_k:
                break
            if r["id"] not in seen:
                selected.append(r)
                seen.add(r["id"])

        selected.sort(key=lambda r: -r["rrf_score"])
        print(f"[SEARCH] dense réservés={len(reserved)} / retenus={len(selected)}")
        return selected
    finally:
        conn.close()
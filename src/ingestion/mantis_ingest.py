import os
import json
from datetime import datetime

import pymysql
import psycopg2
import torch

from colpali_engine.models.qwen2_5 import ColQwen2_5, ColQwen2_5_Processor


def get_rag_conn():
    db_url = os.getenv("RAG_DATABASE_URL", "postgresql://rag:ragpass@db:5432/ragdb")
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    return conn

def get_existing_mantis_ids():
    conn = get_rag_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source_id
                FROM documents
                WHERE source_type = 'mantis'
                """
            )
            rows = cur.fetchall()

        existing_ids = set()

        for row in rows:
            source_id = row[0]
            if source_id and source_id.startswith("MANTIS-"):
                try:
                    existing_ids.add(int(source_id.replace("MANTIS-", "")))
                except ValueError:
                    pass

        return existing_ids
    finally:
        conn.close()

def get_mantis_conn():
    host = os.getenv("MANTIS_DB_HOST", "mantis-testlink.ice.local")
    port = int(os.getenv("MANTIS_DB_PORT", "3306"))
    db = os.getenv("MANTIS_DB_NAME", "m-pccn")
    user = os.getenv("MANTIS_DB_USER", "rag-tickets")
    password = os.getenv("MANTIS_DB_PASSWORD")

    if not password:
        raise RuntimeError("MANTIS_DB_PASSWORD is missing")

    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=db,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        ssl_disabled=True,
    )


def load_model():
    model_name = os.getenv("EMBED_MODEL", "Metric-AI/ColQwen2.5-3b-multilingual-v1.0")
    device = os.getenv("DEVICE", "cpu")

    print(f"[MANTIS] Loading model: {model_name}")
    print(f"[MANTIS] Device: {device}")

    model = ColQwen2_5.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
    ).eval()

    processor = ColQwen2_5_Processor.from_pretrained(model_name)
    return model, processor, device


def pool_embedding(embeddings: torch.Tensor):
    if embeddings.dim() == 3:
        pooled = embeddings.mean(dim=1).squeeze(0)
    elif embeddings.dim() == 2:
        pooled = embeddings.mean(dim=0)
    else:
        raise ValueError(f"Unexpected embedding shape: {tuple(embeddings.shape)}")

    return pooled.detach().cpu().tolist()


def embed_text(text: str, model, processor, device: str):
    batch = processor.process_queries([text]).to(device)

    with torch.no_grad():
        embeddings = model(**batch)

    return pool_embedding(embeddings)


def fetch_mantis_tickets(last_sync=None):
    sql = """
        SELECT
            b.id,
            b.summary,
            FROM_UNIXTIME(b.date_submitted) AS date_submitted,
            FROM_UNIXTIME(b.last_updated) AS last_updated,
            bt.description,
            bt.steps_to_reproduce,
            bt.additional_information,
            GROUP_CONCAT(
                CONCAT(
                    '[Note du ', FROM_UNIXTIME(n.date_submitted), '] ',
                    nt.note
                )
                ORDER BY n.date_submitted ASC
                SEPARATOR '\\n\\n'
            ) AS notes
        FROM mantis_bug_table b
        LEFT JOIN mantis_bug_text_table bt ON bt.id = b.bug_text_id
        LEFT JOIN mantis_bugnote_table n ON n.bug_id = b.id
        LEFT JOIN mantis_bugnote_text_table nt ON nt.id = n.bugnote_text_id
    """

    params = []

    if last_sync is not None:
        sql += """
        WHERE FROM_UNIXTIME(b.last_updated) > %s
        """
        params.append(last_sync)

    sql += """
        GROUP BY
            b.id,
            b.summary,
            b.date_submitted,
            b.last_updated,
            bt.description,
            bt.steps_to_reproduce,
            bt.additional_information
        ORDER BY b.last_updated ASC
    """

    conn = get_mantis_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()


def build_mantis_content(ticket: dict) -> str:
    parts = [
        f"Ticket Mantis: {ticket.get('id')}",
        f"Résumé: {ticket.get('summary') or ''}",
        f"Date création: {ticket.get('date_submitted') or ''}",
        f"Dernière modification: {ticket.get('last_updated') or ''}",
        "",
        f"Description:\n{ticket.get('description') or ''}",
        "",
        f"Étapes pour reproduire:\n{ticket.get('steps_to_reproduce') or ''}",
        "",
        f"Informations complémentaires:\n{ticket.get('additional_information') or ''}",
    ]

    if ticket.get("notes"):
        parts.extend([
            "",
            f"Notes / commentaires:\n{ticket.get('notes')}",
        ])

    return "\n".join(parts).strip()


def insert_mantis_ticket(conn, ticket: dict, content: str, embedding: list):
    source_id = f"MANTIS-{ticket['id']}"
    source_path = f"MANTIS-{ticket['id']}"

    metadata = {
        "mantis_id": ticket.get("id"),
        "summary": ticket.get("summary"),
        "date_submitted": str(ticket.get("date_submitted")) if ticket.get("date_submitted") else None,
        "last_updated": str(ticket.get("last_updated")) if ticket.get("last_updated") else None,
        "source_table": "mantis_bug_table",
    }

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (
                source_type,
                source_id,
                source_path,
                page_num,
                row_num,
                chunk_id,
                content,
                metadata,
                pooled_embedding,
                pccn_version,
                page_image_base64
            )
            VALUES (%s, %s, %s, NULL, NULL, %s, %s, %s::jsonb, %s::vector, NULL, NULL)
            ON CONFLICT (source_type, source_id, chunk_id)
            DO UPDATE SET
                source_path = EXCLUDED.source_path,
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                pooled_embedding = EXCLUDED.pooled_embedding
            """,
            (
                "mantis",
                source_id,
                source_path,
                "main",
                content,
                json.dumps(metadata),
                str(embedding),
            ),
        )


def ingest_mantis(last_sync=None):
    if last_sync is None:
        print("[MANTIS] First sync: fetching all Mantis tickets")
    else:
        print(f"[MANTIS] Incremental sync since {last_sync}")

    tickets = fetch_mantis_tickets(last_sync=last_sync)
    print(f"[MANTIS] {len(tickets)} ticket(s) fetched")
    existing_ids = get_existing_mantis_ids()
    print(f"[MANTIS] {len(existing_ids)} ticket(s) already indexed")
    tickets = [t for t in tickets if int(t["id"]) not in existing_ids]
    print(f"[MANTIS] {len(tickets)} ticket(s) remaining after excluding already indexed tickets")
    if not tickets:
        print("[MANTIS] No tickets to ingest")
        return 0

    model, processor, device = load_model()
    rag_conn = get_rag_conn()

    count = 0

    try:
        for ticket in tickets:
            content = build_mantis_content(ticket)

            if not content.strip():
                print(f"[MANTIS] Skip empty ticket {ticket.get('id')}")
                continue

            embedding = embed_text(content, model, processor, device)

            insert_mantis_ticket(
                conn=rag_conn,
                ticket=ticket,
                content=content,
                embedding=embedding,
            )

            count += 1
            print(f"[MANTIS] Insert OK: MANTIS-{ticket.get('id')}")

        return count

    finally:
        rag_conn.close()
"""Écriture en base, commune aux documents et aux tickets."""
import json

from psycopg2.extras import execute_values

from src.common.db import vector_literal

_COLUMNS = (
    "source_type", "source_id", "source_path", "page_num", "chunk_id", "content",
    "metadata", "embedding", "pccn_version", "tranche", "dicodata_version",
)


def _clean(text):
    # Postgres refuse le caractère NUL dans un TEXT
    return (text or "").replace("\x00", "")


def replace_source(conn, source_type: str, source_id: str, chunks: list[dict]):
    """
    Remplace TOUS les chunks d'une source en une transaction.
    Corrige : les pages fantômes quand un document est ré-uploadé plus court,
    et les anciens chunks 'main' des tickets.
    `conn` doit être en autocommit=False.
    """
    rows = [
        (
            source_type,
            source_id,
            c.get("source_path"),
            c.get("page_num"),
            c["chunk_id"],
            _clean(c["content"]),
            json.dumps(c.get("metadata") or {}, ensure_ascii=False, default=str),
            vector_literal(c["embedding"]),
            c.get("pccn_version"),
            c.get("tranche"),
            c.get("dicodata_version"),
        )
        for c in chunks
    ]
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM documents WHERE source_type = %s AND source_id = %s",
                (source_type, source_id),
            )
            if rows:
                execute_values(
                    cur,
                    f"INSERT INTO documents ({', '.join(_COLUMNS)}) VALUES %s",
                    rows,
                    template="(%s,%s,%s,%s,%s,%s,%s::jsonb,%s::vector,%s,%s,%s)",
                )


def delete_source(conn, source_type: str, source_id: str):
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM documents WHERE source_type = %s AND source_id = %s",
                (source_type, source_id),
            )


def get_cursor(conn, source_name: str):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT sync_cursor FROM ingestion_state WHERE source_name = %s",
            (source_name,),
        )
        row = cur.fetchone()
    conn.commit()
    return row[0] if row else None


def save_cursor(conn, source_name: str, cursor: dict | None):
    """Sauvegardé après chaque lot : un plantage reprend là où il s'est arrêté."""
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO ingestion_state (source_name, sync_cursor, last_successful_sync, updated_at)
                   VALUES (%s, %s::jsonb, now(), now())
                   ON CONFLICT (source_name) DO UPDATE
                      SET sync_cursor = EXCLUDED.sync_cursor,
                          last_successful_sync = now(),
                          updated_at = now()""",
                (source_name, json.dumps(cursor, default=str) if cursor is not None else None),
            )

import os
import psycopg2

from src.ingestion.ticket_ingest import ingest_tickets
from src.ingestion.run import main as ingest_uploaded_documents


def get_rag_conn():
    db_url = os.getenv("RAG_DATABASE_URL", "postgresql://rag:ragpass@db:5432/ragdb")
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    return conn


def get_last_sync(source_name: str):
    conn = get_rag_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT last_successful_sync
                FROM ingestion_state
                WHERE source_name = %s
                """,
                (source_name,),
            )
            result = cur.fetchone()
            return result[0] if result else None
    finally:
        conn.close()


def update_last_sync(source_name: str):
    conn = get_rag_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ingestion_state
                SET last_successful_sync = NOW(),
                    updated_at = NOW()
                WHERE source_name = %s
                """,
                (source_name,),
            )
    finally:
        conn.close()


def main():
    print("[SYNC] Starting weekly sync")

    # 1) Tickets
    last_ticket_sync = get_last_sync("tickets")
    print(f"[SYNC] Last sync for tickets: {last_ticket_sync}")

    ticket_count = ingest_tickets(last_sync=last_ticket_sync)

    update_last_sync("tickets")
    print(f"[SYNC] Tickets sync done ({ticket_count} ticket(s))")

    # 2) Uploaded documents
    print("[SYNC] Starting uploaded documents ingestion")
    ingest_uploaded_documents()

    update_last_sync("uploaded_docs")
    print("[SYNC] Uploaded documents sync done")

    print("[SYNC] Weekly sync finished successfully")


if __name__ == "__main__":
    main()
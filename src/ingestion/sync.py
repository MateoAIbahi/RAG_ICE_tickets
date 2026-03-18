import os
import psycopg2

from src.ingestion.ticket_ingest import ingest_tickets


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
    last_sync = get_last_sync("tickets")
    print(f"[SYNC] Last sync for tickets: {last_sync}")

    count = ingest_tickets(last_sync=last_sync)

    if count >= 0:
        update_last_sync("tickets")
        print("[SYNC] Tickets sync updated successfully")


if __name__ == "__main__":
    main()
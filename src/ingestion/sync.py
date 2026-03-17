import os
import psycopg2
from datetime import datetime


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

if __name__ == "__main__":
    last = get_last_sync("tickets")
    print("Last sync tickets:", last)
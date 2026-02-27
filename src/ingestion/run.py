import os
from datetime import datetime, timezone
import psycopg2

RAG_DATABASE_URL = os.getenv("RAG_DATABASE_URL")  # ex: postgresql://rag:ragpass@db:5432/ragdb
ERP_DATABASE_URL = os.getenv("ERP_DATABASE_URL")  # ex: postgresql://ro:pass@erp:5432/erpdb (pas obligatoire pour le test)

def _connect(url: str, label: str):
    if not url:
        print(f"[{label}] DATABASE_URL not set -> skipping")
        return None
    conn = psycopg2.connect(url)
    conn.autocommit = True
    print(f"[{label}] Connected OK")
    return conn

def main():
    # 1) Connect RAG DB (obligatoire)
    rag = _connect(RAG_DATABASE_URL, "RAG")
    if rag is None:
        raise RuntimeError("RAG_DATABASE_URL is required")

    # 2) Lire l'état de sync
    with rag.cursor() as cur:
        cur.execute(
            "SELECT source, last_sync_ts FROM ingestion_state WHERE source=%s",
            ("erp_sylob",),
        )
        row = cur.fetchone()
        print(f"[RAG] Current state for 'erp_sylob': {row}")

    # 3) Ecrire/mettre à jour un état (test)
    now = datetime.now(timezone.utc)
    with rag.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ingestion_state (source, last_sync_ts)
            VALUES (%s, %s)
            ON CONFLICT (source) DO UPDATE
              SET last_sync_ts = EXCLUDED.last_sync_ts,
                  updated_at = now()
            """,
            ("erp_sylob", now),
        )
    print(f"[RAG] Updated ingestion_state('erp_sylob') -> {now.isoformat()}")

    # 4) Connect ERP DB (optionnel pour l'instant)
    erp = _connect(ERP_DATABASE_URL, "ERP")
    if erp:
        with erp.cursor() as cur:
            cur.execute("SELECT 1;")
            print("[ERP] SELECT 1 OK")
        erp.close()

    rag.close()
    print("[DONE] Batch ingestion MVP finished")

if __name__ == "__main__":
    main()
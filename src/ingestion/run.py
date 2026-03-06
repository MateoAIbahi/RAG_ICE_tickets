import os
from datetime import datetime, timezone
import psycopg2

RAG_DATABASE_URL = os.getenv("RAG_DATABASE_URL")
ERP_DATABASE_URL = os.getenv("ERP_DATABASE_URL")


def _connect(url: str, label: str):
    if not url:
        print(f"[{label}] DATABASE_URL not set -> skipping")
        return None
    conn = psycopg2.connect(url)
    conn.autocommit = True
    print(f"[{label}] Connected OK")
    return conn


def main():
    if os.getenv("PROBE_ONLY") == "1":
        from src.ingestion.probe import run_probe
        run_probe()
        return

    rag = _connect(RAG_DATABASE_URL, "RAG")
    if rag is None:
        raise RuntimeError("RAG_DATABASE_URL is required")

    with rag.cursor() as cur:
        cur.execute(
            "SELECT source, last_sync_ts FROM ingestion_state WHERE source=%s",
            ("erp_sylob",),
        )
        row = cur.fetchone()
        print(f"[RAG] Current state for 'erp_sylob': {row}")

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

    if ERP_DATABASE_URL:
        erp = _connect(ERP_DATABASE_URL, "ERP")
        if erp:
            with erp.cursor() as cur:
                cur.execute("SELECT 1;")
                print("[ERP] SELECT 1 OK")
            erp.close()
    else:
        print("[ERP] Not configured -> skipping")

    rag.close()
    print("[DONE] Batch ingestion MVP finished")


if __name__ == "__main__":
    main()
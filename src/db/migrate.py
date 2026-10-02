"""
Applique les fichiers src/db/migrations/*.sql dans l'ordre, une seule fois chacun.
Appelé au démarrage de l'API et de l'ingestion (verrou consultatif => pas de
course si les deux démarrent en même temps). Lancement manuel :
    python -m src.db.migrate
"""
import os
from pathlib import Path

from src.common.db import get_conn

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
LOCK_ID = 74210001


def migrate():
    dim = os.getenv("EMBED_DIM", "1024")
    conn = get_conn(autocommit=False)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s)", (LOCK_ID,))
            cur.execute(
                """CREATE TABLE IF NOT EXISTS schema_migrations (
                       name TEXT PRIMARY KEY,
                       applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"""
            )
            conn.commit()
            cur.execute("SELECT name FROM schema_migrations")
            done = {r[0] for r in cur.fetchall()}

            for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
                if path.name in done:
                    continue
                print(f"[MIGRATE] Application de {path.name}")
                sql = path.read_text(encoding="utf-8").replace("{{EMBED_DIM}}", dim)
                cur.execute(sql)
                cur.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,))
                conn.commit()
            cur.execute("SELECT pg_advisory_unlock(%s)", (LOCK_ID,))
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
    print("[MIGRATE] OK")

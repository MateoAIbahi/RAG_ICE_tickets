"""
Tickets Mantis (MySQL).

Corrections :
- l'ancien code prenait 100 tickets puis enregistrait NOW() comme date de sync :
  tous les tickets plus anciens non traités étaient perdus pour toujours ;
- `exclude_ids` empêchait toute mise à jour d'un ticket déjà indexé (nouvelle
  note, solution ajoutée...). Supprimé : le remplacement atomique gère les doublons ;
- GROUP_CONCAT est limité à 1024 caractères par défaut dans MySQL : les notes
  étaient tronquées silencieusement ;
- curseur sur (last_updated brut en secondes, id) : pas de problème de fuseau
  horaire entre MySQL et Postgres, pas d'ex aequo perdu entre deux lots ;
- projet et version Mantis conservés en métadonnées (piste pour rattacher les
  tickets à une version PCCN).
"""
import os

import pymysql

from src.common.db import get_conn
from src.common.embedder import get_embedder
from src.ingestion.store import get_cursor, replace_source, save_cursor

SOURCE = "mantis"
BATCH_SIZE = int(os.getenv("MANTIS_BATCH_SIZE", "200"))

MANTIS_QUERY = """
SELECT
    b.id,
    b.summary,
    b.last_updated AS last_updated_raw,
    FROM_UNIXTIME(b.date_submitted) AS date_submitted,
    FROM_UNIXTIME(b.last_updated)   AS last_updated,
    b.version,
    p.name AS project,
    bt.description,
    bt.steps_to_reproduce,
    bt.additional_information,
    GROUP_CONCAT(
        CONCAT('[Note du ', FROM_UNIXTIME(n.date_submitted), '] ', nt.note)
        ORDER BY n.date_submitted ASC
        SEPARATOR '\\n\\n'
    ) AS notes
FROM mantis_bug_table b
LEFT JOIN mantis_project_table p      ON p.id = b.project_id
LEFT JOIN mantis_bug_text_table bt    ON bt.id = b.bug_text_id
LEFT JOIN mantis_bugnote_table n      ON n.bug_id = b.id
LEFT JOIN mantis_bugnote_text_table nt ON nt.id = n.bugnote_text_id
WHERE b.last_updated > %s OR (b.last_updated = %s AND b.id > %s)
GROUP BY b.id, b.summary, b.last_updated, b.date_submitted, b.version, p.name,
         bt.description, bt.steps_to_reproduce, bt.additional_information
ORDER BY b.last_updated ASC, b.id ASC
LIMIT %s
"""


def get_mantis_conn():
    password = os.getenv("MANTIS_DB_PASSWORD")
    if not password:
        return None
    conn = pymysql.connect(
        host=os.getenv("MANTIS_DB_HOST", "mantis-testlink.ice.local"),
        port=int(os.getenv("MANTIS_DB_PORT", "3306")),
        user=os.getenv("MANTIS_DB_USER", "rag-tickets"),
        password=password,
        database=os.getenv("MANTIS_DB_NAME", "m-pccn"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        ssl_disabled=True,
    )
    with conn.cursor() as cur:
        cur.execute("SET SESSION group_concat_max_len = 4194304")
    return conn


def build_header(t: dict) -> str:
    h = f"Ticket Mantis {t['id']} : {t.get('summary') or ''}"
    extra = [x for x in (t.get("project"), t.get("version") and f"version {t['version']}") if x]
    if extra:
        h += f" ({', '.join(extra)})"
    return h


def build_body(t: dict) -> str:
    sections = [
        ("Description", t.get("description")),
        ("Étapes pour reproduire", t.get("steps_to_reproduce")),
        ("Informations complémentaires", t.get("additional_information")),
        ("Notes / commentaires", t.get("notes")),
    ]
    return "\n\n".join(f"{name} :\n{val.strip()}" for name, val in sections if val and val.strip())


def ingest_mantis() -> dict | str:
    mantis = get_mantis_conn()
    if mantis is None:
        return "ignoré (MANTIS_DB_PASSWORD non configuré)"

    rag = get_conn(autocommit=False)
    emb = get_embedder()
    cursor = get_cursor(rag, SOURCE) or {"ts": 0, "id": 0}
    print(f"[MANTIS] Reprise depuis {cursor}")
    stats = {"indexes": 0, "lots": 0}

    try:
        while True:
            with mantis.cursor() as cur:
                cur.execute(MANTIS_QUERY, (cursor["ts"], cursor["ts"], cursor["id"], BATCH_SIZE))
                rows = cur.fetchall()
            if not rows:
                break

            todo = []
            for t in rows:
                header = build_header(t)
                pieces = emb.chunk(build_body(t), header) or [""]
                todo.append((t, header, pieces))

            texts = [f"{h}\n{p}".strip() for _, h, ps in todo for p in ps]
            vectors = iter(emb.embed_documents(texts))

            for t, header, pieces in todo:
                sid = f"MANTIS-{t['id']}"
                chunks = []
                for i, p in enumerate(pieces):
                    chunks.append({
                        "source_path": sid,
                        "chunk_id": f"c{i}",
                        "content": f"{header}\n{p}".strip(),
                        "embedding": next(vectors),
                        "metadata": {
                            "mantis_id": t["id"],
                            "summary": t.get("summary"),
                            "project": t.get("project"),
                            "mantis_version": t.get("version"),
                            "date_submitted": t.get("date_submitted"),
                            "last_updated": t.get("last_updated"),
                            "chunks": len(pieces),
                            "source_table": "mantis_bug_table",
                        },
                    })
                replace_source(rag, "mantis", sid, chunks)
                stats["indexes"] += 1

            last = rows[-1]
            cursor = {"ts": int(last["last_updated_raw"]), "id": int(last["id"])}
            save_cursor(rag, SOURCE, cursor)
            stats["lots"] += 1
            print(f"[MANTIS] Lot {stats['lots']} : {len(rows)} ticket(s), curseur={cursor}")
    finally:
        mantis.close()
        rag.close()

    print(f"[MANTIS] Bilan : {stats}")
    return stats

"""
Tickets Sylob (ERP Postgres, table specif_ticket_mcosup).

Corrections :
- curseur (date de modif, id) = dernier ticket réellement traité, et non NOW()
  -> aucun ticket sauté, reprise exacte après un plantage ;
- traitement par lots (RAM maîtrisée) ;
- tickets invalidés (datefinvalidite renseignée) retirés de l'index ;
- tickets longs découpés en plusieurs chunks (la Solution, dernier champ,
  était la première victime de la troncature).
"""
import os

import psycopg2

from src.common.db import get_conn
from src.common.embedder import get_embedder
from src.ingestion.store import delete_source, get_cursor, replace_source, save_cursor

SOURCE = "tickets"
BATCH_SIZE = int(os.getenv("TICKETS_BATCH_SIZE", "200"))
EPOCH = "1970-01-01 00:00:00"

TICKETS_QUERY = """
SELECT * FROM (
    SELECT
        TIK.id,
        TIK.code,
        TIK.libelle,
        COALESCE(TIK.datemodificationsysteme, TIK.datecreationsysteme, %(epoch)s) AS last_update,
        TIK.datefinvalidite,
        DES.alpha    AS description,
        ANALYS.alpha AS analyse_interne,
        ANALYS2.alpha AS analyse_interne2,
        SOLUS.alpha  AS solution,
        (SELECT ART6.code FROM dm1_article ART6 WHERE EQP.alpha = ART6.id) AS equipement
    FROM specif_ticket_mcosup TIK
    LEFT OUTER JOIN specif_ticket_mcosup_cs DES
        ON TIK.id = DES.id_ticket_mcosup AND DES.clef = 'sup_Ticket_MCOSup_description_du_probleme'
    LEFT OUTER JOIN specif_ticket_mcosup_cs EQP
        ON TIK.id = EQP.id_ticket_mcosup AND EQP.clef = 'sup_Ticket_MCOSup_equipement_concerne'
    LEFT OUTER JOIN specif_ticket_mcosup_cs ANALYS
        ON TIK.id = ANALYS.id_ticket_mcosup AND ANALYS.clef = 'sup_Ticket_MCOSup_analyse_interne'
    LEFT OUTER JOIN specif_ticket_mcosup_cs ANALYS2
        ON TIK.id = ANALYS2.id_ticket_mcosup AND ANALYS2.clef = 'sup_Ticket_MCOSup_analyse_interne_suite'
    LEFT OUTER JOIN specif_ticket_mcosup_cs SOLUS
        ON TIK.id = SOLUS.id_ticket_mcosup AND SOLUS.clef = 'sup_Ticket_MCOSup_solution_apportee'
    WHERE TIK.id NOT LIKE 'DefaultRecord_%%'
) t
WHERE t.last_update > %(ts)s
   OR (t.last_update = %(ts)s AND t.id > %(id)s)
ORDER BY t.last_update, t.id
LIMIT %(limit)s
"""


def get_erp_conn():
    url = os.getenv("ERP_DATABASE_URL")
    if not url:
        return None
    conn = psycopg2.connect(url)
    conn.set_session(readonly=True, autocommit=True)
    return conn


def build_header(t: dict) -> str:
    h = f"Ticket Sylob {t['code'] or t['id']} : {t['libelle'] or ''}"
    if t.get("equipement"):
        h += f" (équipement {t['equipement']})"
    return h


def build_body(t: dict) -> str:
    sections = [
        ("Description", t["description"]),
        ("Analyse interne", t["analyse_interne"]),
        ("Analyse interne (suite)", t["analyse_interne2"]),
        ("Solution", t["solution"]),
    ]
    return "\n\n".join(f"{name} :\n{val.strip()}" for name, val in sections if val and val.strip())


def ingest_tickets() -> dict | str:
    erp = get_erp_conn()
    if erp is None:
        return "ignoré (ERP_DATABASE_URL non configuré)"

    rag = get_conn(autocommit=False)
    emb = get_embedder()
    cursor = get_cursor(rag, SOURCE) or {"ts": EPOCH, "id": ""}
    print(f"[TICKETS] Reprise depuis {cursor}")
    stats = {"indexes": 0, "retires": 0, "lots": 0}

    try:
        while True:
            with erp.cursor() as cur:
                cur.execute(TICKETS_QUERY, {
                    "epoch": EPOCH, "ts": cursor["ts"], "id": cursor["id"], "limit": BATCH_SIZE,
                })
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            if not rows:
                break

            todo = []  # (ticket, header, morceaux)
            for t in rows:
                if t["datefinvalidite"] is not None:
                    delete_source(rag, "ticket", str(t["id"]))
                    stats["retires"] += 1
                    continue
                header = build_header(t)
                pieces = emb.chunk(build_body(t), header) or [""]
                todo.append((t, header, pieces))

            texts = [f"{h}\n{p}".strip() for _, h, ps in todo for p in ps]
            vectors = iter(emb.embed_documents(texts)) if texts else iter(())

            for t, header, pieces in todo:
                chunks = []
                for i, p in enumerate(pieces):
                    chunks.append({
                        "source_path": t["code"] or str(t["id"]),
                        "chunk_id": f"c{i}",
                        "content": f"{header}\n{p}".strip(),
                        "embedding": next(vectors),
                        "metadata": {
                            "ticket_code": t["code"],
                            "libelle": t["libelle"],
                            "equipement": t["equipement"],
                            "last_update": t["last_update"],
                            "chunks": len(pieces),
                            "source_table": "specif_ticket_mcosup",
                        },
                    })
                replace_source(rag, "ticket", str(t["id"]), chunks)
                stats["indexes"] += 1

            last = rows[-1]
            cursor = {"ts": str(last["last_update"]), "id": str(last["id"])}
            save_cursor(rag, SOURCE, cursor)
            stats["lots"] += 1
            print(f"[TICKETS] Lot {stats['lots']} : {len(rows)} ticket(s), curseur={cursor}")
    finally:
        erp.close()
        rag.close()

    print(f"[TICKETS] Bilan : {stats}")
    return stats

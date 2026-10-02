"""
Synchronisation hebdomadaire (lancée le vendredi soir par deploy/run_ingestion.sh).

    python -m src.ingestion.sync               # tout
    python -m src.ingestion.sync --docs-only   # uniquement les documents en attente

Chaque source est isolée : si Mantis est injoignable, les tickets Sylob et les
documents sont quand même traités. Code de sortie 1 si une source a échoué,
pour que l'échec soit visible dans `systemctl status` / les logs du cron.
"""
import sys
import time
import traceback

from src.common.db import get_conn, vector_literal
from src.common.embedder import get_embedder
from src.db.migrate import migrate
from src.ingestion.documents import OCR_MIN_CHARS, ingest_documents, ocr_image_b64
from src.ingestion.mantis_ingest import ingest_mantis
from src.ingestion.store import save_cursor
from src.ingestion.ticket_ingest import ingest_tickets

SYNC_LOCK_ID = 74210002


def reembed_missing(batch: int = 128) -> dict:
    """
    Calcule l'embedding des lignes qui n'en ont pas : essentiellement les PDF
    déjà en base avant la migration (leurs fichiers d'origine ont été supprimés
    par l'ancienne ingestion, on repart donc du texte stocké).

    Les anciennes pages sans texte ("[PAGE_IMAGE_ONLY]") n'étaient trouvables
    que par l'image ColQwen : on tente un OCR sur l'image stockée.
    """
    conn = get_conn(autocommit=False)
    emb = get_embedder()
    stats = {"lignes": 0, "ocr_reussis": 0, "ocr_vides": 0}
    try:
        while True:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT id, content, page_image_base64, source_path, page_num, pccn_version
                         FROM documents
                        WHERE embedding IS NULL AND deleted_at IS NULL
                        ORDER BY id LIMIT %s""",
                    (batch,),
                )
                rows = cur.fetchall()
            conn.commit()
            if not rows:
                break

            contents, ocr_updates = [], {}
            for rid, content, img, path, page, version in rows:
                if content.startswith("[PAGE_IMAGE_ONLY]") and img:
                    text = ocr_image_b64(img)
                    if len(text) >= OCR_MIN_CHARS:
                        content = (f"Document : {path} | PCCN {version or '?'} | page {page} "
                                   f"(OCR basse résolution)\n{text}")
                        ocr_updates[rid] = content
                        stats["ocr_reussis"] += 1
                    else:
                        stats["ocr_vides"] += 1
                contents.append(content)

            vectors = emb.embed_documents(contents)
            with conn:
                with conn.cursor() as cur:
                    for (rid, *_), v in zip(rows, vectors):
                        if rid in ocr_updates:
                            cur.execute(
                                """UPDATE documents
                                      SET content = %s, embedding = %s::vector,
                                          metadata = metadata || '{"ocr_legacy": true}'::jsonb
                                    WHERE id = %s""",
                                (ocr_updates[rid], vector_literal(v), rid),
                            )
                        else:
                            cur.execute(
                                "UPDATE documents SET embedding = %s::vector WHERE id = %s",
                                (vector_literal(v), rid),
                            )
            stats["lignes"] += len(rows)
            print(f"[REEMBED] {stats}")
    finally:
        conn.close()
    return stats


def _docs_step():
    stats = ingest_documents()
    conn = get_conn(autocommit=False)
    try:
        save_cursor(conn, "uploaded_docs", None)
    finally:
        conn.close()
    return stats


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    t0 = time.time()
    print("[SYNC] Démarrage")
    migrate()

    lock_conn = get_conn()
    with lock_conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (SYNC_LOCK_ID,))
        if not cur.fetchone()[0]:
            print("[SYNC] Une synchronisation est déjà en cours, abandon.")
            return 0

    if "--docs-only" in argv:
        steps = [("documents", _docs_step)]
    else:
        steps = [
            # Documents d'abord : c'est ce qu'ICE attend en priorité
            ("documents", _docs_step),
            ("tickets_sylob", ingest_tickets),
            ("tickets_mantis", ingest_mantis),
            ("rattrapage_embeddings", reembed_missing),
        ]

    results, failed = {}, False
    for name, fn in steps:
        print(f"\n[SYNC] ===== {name} =====")
        t = time.time()
        try:
            results[name] = f"OK {fn()} ({time.time() - t:.0f}s)"
        except Exception as e:
            failed = True
            traceback.print_exc()
            results[name] = f"ÉCHEC {e!r}"

    lock_conn.close()  # libère le verrou
    print("\n[SYNC] ===== Bilan =====")
    for name, res in results.items():
        print(f"  {name:<24} {res}")
    print(f"[SYNC] Terminé en {(time.time() - t0) / 60:.1f} min")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

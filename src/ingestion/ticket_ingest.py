import os
import json
import psycopg2
import torch

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor


TICKETS_QUERY_ALL = """
select
    TIK.id,
    TIK.code,
    TIK.libelle,
    COALESCE(TIK.datemodificationsysteme, TIK.datecreationsysteme) as last_update,
    DES.alpha as description,
    ANALYS.alpha as analyse_interne,
    ANALYS2.alpha as analyse_interne2,
    SOLUS.alpha as solution,
    (select ART6.code from dm1_article ART6 where EQP.alpha=ART6.id) as equipement
from specif_ticket_mcosup TIK
left outer join specif_ticket_mcosup_cs DES
    on TIK.id=DES.id_ticket_mcosup
   and DES.clef='sup_Ticket_MCOSup_description_du_probleme'
left outer join specif_ticket_mcosup_cs EQP
    on TIK.id=EQP.id_ticket_mcosup
   and EQP.clef='sup_Ticket_MCOSup_equipement_concerne'
left outer join specif_ticket_mcosup_cs ANALYS
    on TIK.id=ANALYS.id_ticket_mcosup
   and ANALYS.clef='sup_Ticket_MCOSup_analyse_interne'
left outer join specif_ticket_mcosup_cs ANALYS2
    on TIK.id=ANALYS2.id_ticket_mcosup
   and ANALYS2.clef='sup_Ticket_MCOSup_analyse_interne_suite'
left outer join specif_ticket_mcosup_cs SOLUS
    on TIK.id=SOLUS.id_ticket_mcosup
   and SOLUS.clef='sup_Ticket_MCOSup_solution_apportee'
where TIK.id not like 'DefaultRecord_%%'
  and TIK.datefinvalidite is null
order by last_update asc
"""

TICKETS_QUERY_SINCE = """
select
    TIK.id,
    TIK.code,
    TIK.libelle,
    COALESCE(TIK.datemodificationsysteme, TIK.datecreationsysteme) as last_update,
    DES.alpha as description,
    ANALYS.alpha as analyse_interne,
    ANALYS2.alpha as analyse_interne2,
    SOLUS.alpha as solution,
    (select ART6.code from dm1_article ART6 where EQP.alpha=ART6.id) as equipement
from specif_ticket_mcosup TIK
left outer join specif_ticket_mcosup_cs DES
    on TIK.id=DES.id_ticket_mcosup
   and DES.clef='sup_Ticket_MCOSup_description_du_probleme'
left outer join specif_ticket_mcosup_cs EQP
    on TIK.id=EQP.id_ticket_mcosup
   and EQP.clef='sup_Ticket_MCOSup_equipement_concerne'
left outer join specif_ticket_mcosup_cs ANALYS
    on TIK.id=ANALYS.id_ticket_mcosup
   and ANALYS.clef='sup_Ticket_MCOSup_analyse_interne'
left outer join specif_ticket_mcosup_cs ANALYS2
    on TIK.id=ANALYS2.id_ticket_mcosup
   and ANALYS2.clef='sup_Ticket_MCOSup_analyse_interne_suite'
left outer join specif_ticket_mcosup_cs SOLUS
    on TIK.id=SOLUS.id_ticket_mcosup
   and SOLUS.clef='sup_Ticket_MCOSup_solution_apportee'
where TIK.id not like 'DefaultRecord_%%'
  and TIK.datefinvalidite is null
  and COALESCE(TIK.datemodificationsysteme, TIK.datecreationsysteme) > %s
order by last_update asc
"""


def get_rag_conn():
    db_url = os.getenv("RAG_DATABASE_URL", "postgresql://rag:ragpass@db:5432/ragdb")
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    return conn


def get_erp_conn():
    db_url = os.getenv("ERP_DATABASE_URL")
    if not db_url:
        raise ValueError("ERP_DATABASE_URL is not defined")
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    return conn


def load_model():
    model_name = os.getenv("EMBED_MODEL", "Metric-AI/ColQwen2.5-3b-multilingual-v1.0")
    device = os.getenv("DEVICE", "cpu")

    print(f"[TICKETS] Loading model: {model_name}")
    print(f"[TICKETS] Device: {device}")

    model = ColQwen2_5.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
    ).eval()

    processor = ColQwen2_5_Processor.from_pretrained(model_name)
    return model, processor, device


def pool_embedding(embeddings: torch.Tensor):
    if embeddings.dim() == 3:
        pooled = embeddings.mean(dim=1).squeeze(0)
    elif embeddings.dim() == 2:
        pooled = embeddings.mean(dim=0)
    else:
        raise ValueError(f"Unexpected embedding shape: {tuple(embeddings.shape)}")

    return pooled.detach().cpu().tolist()


def embed_text(text: str, model, processor, device: str):
    batch = processor.process_queries([text]).to(device)

    with torch.no_grad():
        embeddings = model(**batch)

    return pool_embedding(embeddings)


def fetch_tickets(last_sync=None):
    conn = get_erp_conn()
    try:
        with conn.cursor() as cur:
            if last_sync is None:
                print("[TICKETS] First sync: fetching all tickets")
                cur.execute(TICKETS_QUERY_ALL)
            else:
                print(f"[TICKETS] Incremental sync since {last_sync}")
                cur.execute(TICKETS_QUERY_SINCE, (last_sync,))

            rows = cur.fetchall()

        tickets = []
        for row in rows:
            tickets.append({
                "ticket_id": row[0],
                "code": row[1],
                "libelle": row[2],
                "last_update": row[3],
                "description": row[4],
                "analyse_interne": row[5],
                "analyse_interne2": row[6],
                "solution": row[7],
                "equipement": row[8],
            })

        print(f"[TICKETS] {len(tickets)} ticket(s) fetched")
        return tickets
    finally:
        conn.close()


def build_ticket_content(ticket: dict) -> str:
    parts = [
        f"Code ticket: {ticket['code'] or ''}",
        f"Libellé: {ticket['libelle'] or ''}",
        f"Équipement: {ticket['equipement'] or ''}",
        f"Description: {ticket['description'] or ''}",
        f"Analyse interne: {ticket['analyse_interne'] or ''}",
        f"Analyse interne suite: {ticket['analyse_interne2'] or ''}",
        f"Solution: {ticket['solution'] or ''}",
    ]
    return "\n".join(parts).strip()


def insert_ticket_document(conn, ticket: dict, content: str, pooled_embedding: list):
    metadata = {
        "ticket_code": ticket["code"],
        "libelle": ticket["libelle"],
        "equipement": ticket["equipement"],
        "last_update": ticket["last_update"].isoformat() if ticket["last_update"] else None,
        "source_table": "specif_ticket_mcosup",
    }

    source_id = ticket["ticket_id"]
    source_path = ticket["code"]
    chunk_id = "main"

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (
                source_type,
                source_id,
                source_path,
                page_num,
                row_num,
                chunk_id,
                content,
                metadata,
                pooled_embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
            ON CONFLICT (source_type, source_id, chunk_id)
            DO UPDATE SET
                source_path = EXCLUDED.source_path,
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                pooled_embedding = EXCLUDED.pooled_embedding
            """,
            (
                "ticket",
                source_id,
                source_path,
                None,
                None,
                chunk_id,
                content,
                json.dumps(metadata, ensure_ascii=False),
                str(pooled_embedding),
            ),
        )


def ingest_tickets(last_sync=None):
    tickets = fetch_tickets(last_sync=last_sync)

    if not tickets:
        print("[TICKETS] No tickets to ingest")
        return 0

    model, processor, device = load_model()
    rag_conn = get_rag_conn()

    inserted = 0
    try:
        for ticket in tickets:
            content = build_ticket_content(ticket)
            pooled_embedding = embed_text(content, model, processor, device)

            insert_ticket_document(
                conn=rag_conn,
                ticket=ticket,
                content=content,
                pooled_embedding=pooled_embedding,
            )

            inserted += 1
            print(f"[TICKETS] Insert/Update OK: {ticket['code']}")
    finally:
        rag_conn.close()

    print(f"[TICKETS] {inserted} ticket(s) ingested")
    return inserted
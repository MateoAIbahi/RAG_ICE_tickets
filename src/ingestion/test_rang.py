import os, psycopg2, torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

MODEL_NAME = os.getenv("EMBED_MODEL", "Metric-AI/ColQwen2.5-3b-multilingual-v1.0")
DB_URL = os.getenv("RAG_DATABASE_URL", "postgresql://rag:ragpass@db:5432/ragdb")

model = ColQwen2_5.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, device_map="cpu").eval()
processor = ColQwen2_5_Processor.from_pretrained(MODEL_NAME)

def embed(text):
    batch = processor.process_queries([text]).to("cpu")
    with torch.no_grad():
        emb = model(**batch)
    return emb.mean(dim=1).squeeze(0).detach().cpu().tolist()

QUESTIONS = [
    ("courte", "Quelle est la limite d'occurrence par barre HTB ?"),
    ("longue", "Hors borne sur la barre HTB3. Le client souhaite connaitre la limite d'occurence par barre HTB."),
]
CIBLE = "ICEPCCN-21105"

conn = psycopg2.connect(DB_URL)
for label, q in QUESTIONS:
    v = str(embed(q))
    with conn.cursor() as cur:
        cur.execute("""
            SELECT source_path, pooled_embedding <=> %s::vector AS d
            FROM documents
            WHERE pooled_embedding IS NOT NULL AND source_type = 'ticket'
            ORDER BY d LIMIT 200
        """, (v,))
        rows = cur.fetchall()
    rang = next((i for i, r in enumerate(rows, 1) if r[0] == CIBLE), None)
    print(f"\n--- {label} : {q}")
    print(f"Rang de {CIBLE} : {rang if rang else '>200'}")
    print("Top 5 :", [(r[0], round(r[1], 4)) for r in rows[:5]])
conn.close()
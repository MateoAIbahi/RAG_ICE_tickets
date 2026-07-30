cat > src/ingestion/test_pdf.py << 'EOF'
import os, psycopg2, torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

MODEL_NAME = os.getenv("EMBED_MODEL", "Metric-AI/ColQwen2.5-3b-multilingual-v1.0")
DB_URL = os.getenv("RAG_DATABASE_URL", "postgresql://rag:ragpass@db:5432/ragdb")

conn = psycopg2.connect(DB_URL)

print("=== INVENTAIRE DES PDF ===")
with conn.cursor() as cur:
    cur.execute("""
        SELECT source_path, pccn_version, count(*), avg(length(content))::int
        FROM documents WHERE source_type = 'pdf'
        GROUP BY 1, 2 ORDER BY 1
    """)
    for path, ver, pages, taille in cur.fetchall():
        print(f"  {str(path)[:55]:55} | version={str(ver):4} | {pages:4} pages | texte moy={taille}")

model = ColQwen2_5.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, device_map="cpu").eval()
processor = ColQwen2_5_Processor.from_pretrained(MODEL_NAME)

def embed(text):
    batch = processor.process_queries([text]).to("cpu")
    with torch.no_grad():
        emb = model(**batch)
    return emb.mean(dim=1).squeeze(0).detach().cpu().tolist()

TESTS = [
    ("APT", "Comment fonctionne l'APT ?", "V3"),
    ("TR313", "Que faut-il faire quand sur le TR313, non detection de terre resistante lors d'un defaut HTA sur le reseau.", None),
]

for label, q, version in TESTS:
    v = str(embed(q))
    sql = """
        SELECT source_path, page_num, pooled_embedding <=> %s::vector AS d
        FROM documents
        WHERE pooled_embedding IS NOT NULL AND source_type = 'pdf'
    """
    params = [v]
    if version:
        sql += " AND (pccn_version = %s OR pccn_version IS NULL)"
        params.append(version)
    sql += " ORDER BY d LIMIT 10"

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    print(f"\n=== {label} (filtre version={version}) : {q}")
    for i, (path, page, d) in enumerate(rows, 1):
        print(f"  {i:2}. {str(path)[:50]:50} p.{page} — {d:.4f}")

conn.close()
EOF
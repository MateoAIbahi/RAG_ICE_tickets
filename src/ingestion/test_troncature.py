import os
import psycopg2
from colpali_engine.models import ColQwen2_5_Processor

MODEL_NAME = os.getenv("EMBED_MODEL", "Metric-AI/ColQwen2.5-3b-multilingual-v1.0")
DB_URL = os.getenv("RAG_DATABASE_URL", "postgresql://rag:ragpass@db:5432/ragdb")

conn = psycopg2.connect(DB_URL)
with conn.cursor() as cur:
    cur.execute("""
        SELECT source_path, content
        FROM documents
        WHERE source_type = 'ticket'
        ORDER BY length(content) DESC
        LIMIT 1
    """)
    source_path, content = cur.fetchone()
conn.close()

print("=== TICKET :", source_path)
print("Longueur du contenu :", len(content), "caractères")
print()

processor = ColQwen2_5_Processor.from_pretrained(MODEL_NAME)
batch = processor.process_queries([content])

print("Shape des tokens :", batch["input_ids"].shape)
print("Tokens réels (hors padding) :", batch["attention_mask"].sum().item())
print()
print("=== TEXTE RÉELLEMENT ENCODÉ ===")
print(processor.tokenizer.decode(batch["input_ids"][0]))
-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- Table documents (chunks)
CREATE TABLE IF NOT EXISTS documents (
  id BIGSERIAL PRIMARY KEY,

  -- provenance
  source_type TEXT NOT NULL,              -- 'ticket_csv', 'pdf', etc.
  source_id   TEXT,                       -- id ticket, nom fichier, etc.
  source_path TEXT,                       -- chemin fichier si besoin
  page_num    INTEGER,                    -- pour PDF
  row_num     INTEGER,                    -- pour CSV
  chunk_id    TEXT,                       -- ex: "ticket123#chunk04"

  -- contenu
  content     TEXT NOT NULL,

  -- métadonnées
  author      TEXT,
  doc_date    TIMESTAMPTZ,
  metadata    JSONB DEFAULT '{}'::jsonb,

  -- embedding
  embedding   VECTOR(1024),               -- ⚠️ on ajustera la dimension à TON modèle
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes utiles
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents (source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_documents_metadata_gin ON documents USING GIN (metadata);

-- Index vectoriel (à activer quand on aura confirmé le type d’index + dimensions + volume)
-- CREATE INDEX IF NOT EXISTS idx_documents_embedding
-- ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
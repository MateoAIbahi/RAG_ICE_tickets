CREATE EXTENSION IF NOT EXISTS vector;

-- Chunks + metadata
CREATE TABLE IF NOT EXISTS documents (
  id BIGSERIAL PRIMARY KEY,
  source_type TEXT NOT NULL,
  source_id   TEXT,
  source_path TEXT,
  page_num    INTEGER,
  row_num     INTEGER,
  chunk_id    TEXT,
  content     TEXT NOT NULL,
  metadata    JSONB DEFAULT '{}'::jsonb,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

  -- 1 vecteur "pooled" (pour préfiltrer vite)
  pooled_embedding VECTOR(128)
);

CREATE INDEX IF NOT EXISTS idx_documents_source ON documents (source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_documents_metadata_gin ON documents USING GIN (metadata);

-- Multi-vecteurs ColQwen (late interaction)
CREATE TABLE IF NOT EXISTS document_vectors (
  document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  vec_idx     INTEGER NOT NULL,
  embedding   VECTOR(128) NOT NULL,
  PRIMARY KEY (document_id, vec_idx)
);


-- Index ANN sur pooled_embedding
-- CREATE INDEX IF NOT EXISTS idx_documents_pooled_ann
-- ON documents USING hnsw (pooled_embedding vector_cosine_ops);
-- Schéma v2. Fonctionne sur une base vide ET sur la base de prod existante
-- (dont plusieurs colonnes avaient été ajoutées à la main, hors dépôt).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id          BIGSERIAL PRIMARY KEY,
  source_type TEXT NOT NULL,          -- pdf | ticket | mantis
  source_id   TEXT,
  source_path TEXT,
  page_num    INTEGER,
  row_num     INTEGER,
  chunk_id    TEXT,
  content     TEXT NOT NULL,
  metadata    JSONB DEFAULT '{}'::jsonb,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE documents ADD COLUMN IF NOT EXISTS pccn_version     TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS tranche          TEXT;   -- AUTO | TRA | RAME | NULL = commun
ALTER TABLE documents ADD COLUMN IF NOT EXISTS dicodata_version TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS deleted_at       TIMESTAMPTZ;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding        vector({{EMBED_DIM}});
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('french', coalesce(content, ''))) STORED;

UPDATE documents SET metadata = '{}'::jsonb WHERE metadata IS NULL;

-- Les PDF déjà en base ont un source_id basé sur le seul nom de fichier
-- (collisions entre versions PCCN). On les marque "legacy" : ils restent
-- interrogeables et sont retirés automatiquement quand le même document est
-- réingéré avec le nouvel identifiant.
UPDATE documents
   SET metadata = metadata || '{"legacy": true}'::jsonb
 WHERE source_type = 'pdf';

-- Anciens vecteurs ColQwen (128 dim, moyennés) : incompatibles avec le nouveau
-- modèle, ils ne peuvent pas cohabiter avec ceux de bge-m3.
ALTER TABLE documents DROP COLUMN IF EXISTS pooled_embedding;
DROP TABLE IF EXISTS document_vectors;

-- page_image_base64 est CONSERVÉE : c'est la seule trace des anciennes pages
-- scannées (fichiers d'origine supprimés par l'ancienne ingestion). L'étape de
-- rattrapage tente un OCR dessus. Colonne à supprimer dans une migration
-- ultérieure, une fois les documents concernés renvoyés par ICE.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS page_image_base64 TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_source_chunk
  ON documents (source_type, source_id, chunk_id);
CREATE INDEX IF NOT EXISTS idx_documents_source       ON documents (source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_documents_metadata_gin ON documents USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_documents_tsv          ON documents USING GIN (content_tsv);
CREATE INDEX IF NOT EXISTS idx_documents_pccn         ON documents (pccn_version);
-- Pas d'index HNSW volontairement : avec les filtres (version, tranche, source)
-- un index ANN peut renvoyer moins de candidats que demandé. Le scan exact
-- reste rapide à cette volumétrie (quelques dizaines de milliers de chunks).

CREATE TABLE IF NOT EXISTS ingestion_state (
  source_name          TEXT PRIMARY KEY,
  last_successful_sync TIMESTAMPTZ,
  updated_at           TIMESTAMPTZ DEFAULT now()
);
ALTER TABLE ingestion_state ADD COLUMN IF NOT EXISTS sync_cursor JSONB;

-- Nouveau modèle => resynchronisation complète des tickets.
UPDATE ingestion_state SET sync_cursor = NULL, last_successful_sync = NULL;
INSERT INTO ingestion_state (source_name)
VALUES ('tickets'), ('mantis'), ('uploaded_docs')
ON CONFLICT (source_name) DO NOTHING;

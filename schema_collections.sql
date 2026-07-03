-- Tier 2: document collections (folders) + scoped uploads.
-- Run once:  psql regintel -f schema_collections.sql

-- A collection is a named folder of uploaded documents, owned by a user.
CREATE TABLE IF NOT EXISTS collections (
    id         SERIAL PRIMARY KEY,
    user_id    TEXT NOT NULL,
    name       TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_collections_user ON collections(user_id);

-- Scope chunks. NULL collection_id = the shared global reference corpus
-- (FDA/eCFR/EMA/MHRA). A non-null collection_id marks a user-uploaded chunk.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS collection_id INTEGER;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS user_id TEXT;
CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks(collection_id);

-- A project references collections (many-to-many).
CREATE TABLE IF NOT EXISTS project_collections (
    project_id    INTEGER REFERENCES projects(id)    ON DELETE CASCADE,
    collection_id INTEGER REFERENCES collections(id) ON DELETE CASCADE,
    PRIMARY KEY (project_id, collection_id)
);

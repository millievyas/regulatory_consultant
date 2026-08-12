-- Storage connections (Tier 2+): one row per client-authorised storage account
-- (Google Drive, SharePoint, Box, ...), linked to a collection. The actual files
-- are NEVER stored here — only their embedded chunks land in `chunks`, scoped by
-- collection_id + user_id exactly like manual uploads.
-- Run once:  psql regintel -f schema_connections.sql

CREATE TABLE IF NOT EXISTS connections (
    id            SERIAL PRIMARY KEY,
    user_id       TEXT NOT NULL,
    collection_id INTEGER REFERENCES collections(id) ON DELETE CASCADE,
    provider      TEXT,                 -- e.g. 'google_drive', 'sharepoint', 'box'
    token         TEXT,                 -- vendor connection token — STORE ENCRYPTED AT REST
    cursor        TEXT,                 -- last-sync marker for incremental pulls
    last_synced   TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_connections_collection ON connections(collection_id);
CREATE INDEX IF NOT EXISTS idx_connections_user ON connections(user_id);

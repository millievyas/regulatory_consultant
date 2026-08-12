-- Warning-letter metadata (the structured/analytics layer).
-- One row per FDA warning letter — enables COUNT / filter / trend / ranking
-- questions that semantic search cannot answer. Bodies still live in `chunks`.
-- Run once:  psql regintel -f schema_letters.sql

CREATE TABLE IF NOT EXISTS warning_letters (
    id           SERIAL PRIMARY KEY,
    company      TEXT,
    office       TEXT,
    subject      TEXT,
    issue_date   DATE,
    year         INTEGER,
    category     TEXT,             -- reserved for phase-3 violation-type classification
    url          TEXT UNIQUE,      -- dedup key
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_wl_year       ON warning_letters(year);
CREATE INDEX IF NOT EXISTS idx_wl_office      ON warning_letters(office);
CREATE INDEX IF NOT EXISTS idx_wl_issue_date  ON warning_letters(issue_date);

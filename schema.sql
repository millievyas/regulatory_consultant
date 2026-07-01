-- Application schema for R Path AI (Tier 1: per-user projects + consultations).
-- Run once:  psql regintel -f schema.sql
--
-- user_id holds the Supabase user id (a UUID string) of the authenticated user.
-- The shared `chunks` knowledge base is global and intentionally NOT scoped here.

CREATE TABLE IF NOT EXISTS projects (
    id              SERIAL PRIMARY KEY,
    user_id         TEXT NOT NULL,
    name            TEXT NOT NULL,
    client          TEXT,
    region          TEXT,
    drug_type       TEXT,
    submission_type TEXT,
    agencies        TEXT,                       -- comma-separated, e.g. "FDA, EMA"
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_id);

CREATE TABLE IF NOT EXISTS consultations (
    id          SERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL,
    project_id  INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    question    TEXT NOT NULL,
    answer      TEXT,
    agents      TEXT,                           -- agents that ran, e.g. "regulatory, quality"
    metrics     JSONB,                          -- {latency, cost, prompt_tokens, completion_tokens}
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_consult_user    ON consultations(user_id);
CREATE INDEX IF NOT EXISTS idx_consult_project ON consultations(project_id);

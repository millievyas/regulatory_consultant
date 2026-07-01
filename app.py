"""FastAPI backend for R Path AI.

Run:  python -m uvicorn app:app --reload
Then open http://127.0.0.1:8000

Auth: each request to a protected endpoint must carry the user's Supabase access
token (Authorization: Bearer <token>). We validate it against Supabase's auth API
and use the returned user id to scope all project/consultation data.
"""

import os
import re
import json
from typing import Optional

import requests
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from agents import coordinate

load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

app = FastAPI(title="R Path AI")


def db():
    return psycopg2.connect(dbname="regintel")


# ------------------------------------------------------------------ auth
def current_user(authorization: str = Header(None)) -> str:
    """Validate the Supabase access token and return the user id."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        r = requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_ANON_KEY},
            timeout=10,
        )
    except Exception:
        raise HTTPException(status_code=503, detail="Auth service unreachable")
    if r.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return r.json()["id"]


# ------------------------------------------------------------------ models
class AskRequest(BaseModel):
    question: str
    history: list = []
    project_id: Optional[int] = None


class ProjectCreate(BaseModel):
    name: str
    client: str = ""
    region: str = ""
    drug_type: str = ""
    submission_type: str = ""
    agencies: str = ""


# ------------------------------------------------------------------ static + config
@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/config")
def config():
    """Public values the browser needs to initialise the Supabase client."""
    return {"supabase_url": SUPABASE_URL, "supabase_anon_key": SUPABASE_ANON_KEY}


# ------------------------------------------------------------------ corpus (shared, public)
@app.get("/overview")
def overview():
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM chunks")
    total_chunks = cur.fetchone()[0]
    cur.execute("SELECT count(DISTINCT COALESCE(company, source_file)) "
                "FROM chunks WHERE source IS NOT NULL")
    total_docs = cur.fetchone()[0]
    cur.execute("SELECT source, count(*) FROM chunks WHERE source IS NOT NULL "
                "GROUP BY source ORDER BY count(*) DESC")
    sources = [{"source": s, "count": c} for s, c in cur.fetchall()]
    cur.execute("SELECT doc_type, count(*) FROM chunks WHERE doc_type IS NOT NULL "
                "GROUP BY doc_type ORDER BY count(*) DESC")
    doc_types = [{"doc_type": d, "count": c} for d, c in cur.fetchall()]
    cur.close()
    conn.close()
    return {"total_chunks": total_chunks, "total_docs": total_docs,
            "sources": sources, "doc_types": doc_types}


@app.get("/documents")
def documents():
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        SELECT COALESCE(company, source_file) AS title, source, doc_type,
               count(*) AS chunks, max(url) AS url
        FROM chunks WHERE source IS NOT NULL
        GROUP BY COALESCE(company, source_file), source, doc_type
        ORDER BY source, title
    """)
    docs = [{"title": t, "source": s, "doc_type": dt, "chunks": c, "url": u}
            for t, s, dt, c, u in cur.fetchall()]
    cur.close()
    conn.close()
    return {"documents": docs}


# ------------------------------------------------------------------ projects (per user)
@app.get("/projects")
def list_projects(user: str = Depends(current_user)):
    conn = db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT p.*,
            (SELECT count(*) FROM consultations c WHERE c.project_id = p.id) AS consultations
        FROM projects p WHERE p.user_id = %s ORDER BY p.created_at DESC
    """, (user,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {"projects": [dict(r) for r in rows]}


@app.post("/projects")
def create_project(body: ProjectCreate, user: str = Depends(current_user)):
    conn = db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        INSERT INTO projects (user_id, name, client, region, drug_type, submission_type, agencies)
        VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING *
    """, (user, body.name, body.client, body.region, body.drug_type,
          body.submission_type, body.agencies))
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return dict(row)


@app.get("/projects/{pid}")
def get_project(pid: int, user: str = Depends(current_user)):
    conn = db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM projects WHERE id = %s AND user_id = %s", (pid, user))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")
    return dict(row)


@app.delete("/projects/{pid}")
def delete_project(pid: int, user: str = Depends(current_user)):
    conn = db()
    cur = conn.cursor()
    cur.execute("DELETE FROM projects WHERE id = %s AND user_id = %s", (pid, user))
    deleted = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"deleted": True}


# ------------------------------------------------------------------ consultations (per user)
@app.get("/consultations")
def list_consultations(project_id: int, user: str = Depends(current_user)):
    conn = db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT * FROM consultations
        WHERE project_id = %s AND user_id = %s ORDER BY created_at DESC
    """, (project_id, user))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {"consultations": [dict(r) for r in rows]}


@app.post("/ask")
def ask(req: AskRequest, user: str = Depends(current_user)):
    answer, metrics = coordinate(req.question, req.history)
    agent_names = re.findall(r"^###\s+(.+?)\s+Agent", answer, re.M)

    conn = db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO consultations (user_id, project_id, question, answer, agents, metrics)
        VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
    """, (user, req.project_id, req.question, answer,
          ", ".join(agent_names), psycopg2.extras.Json(metrics)))
    cid = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return {"answer": answer, "metrics": metrics, "consultation_id": cid}


@app.get("/me/overview")
def me_overview(user: str = Depends(current_user)):
    """Per-user dashboard stats: project/consultation counts + recent activity."""
    conn = db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT count(*) AS n FROM projects WHERE user_id = %s", (user,))
    projects = cur.fetchone()["n"]
    cur.execute("SELECT count(*) AS n FROM consultations WHERE user_id = %s", (user,))
    consultations = cur.fetchone()["n"]
    cur.execute("""
        SELECT c.id, c.question, c.agents, c.created_at, c.project_id, p.name AS project_name
        FROM consultations c
        LEFT JOIN projects p ON p.id = c.project_id
        WHERE c.user_id = %s
        ORDER BY c.created_at DESC LIMIT 6
    """, (user,))
    recent = [dict(r) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return {"projects": projects, "consultations": consultations, "recent": recent}

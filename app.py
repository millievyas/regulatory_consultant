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
from fastapi import FastAPI, Depends, Header, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import pymupdf
from agents import coordinate
from ingest import chunk_text, embed_chunks

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


class CollectionCreate(BaseModel):
    name: str


class AttachCollections(BaseModel):
    collection_ids: list = []


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


def project_collection_ids(project_id, user):
    """Return the user's own collections attached to their project (or None)."""
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM projects WHERE id = %s AND user_id = %s", (project_id, user))
    if not cur.fetchone():
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Project not found")
    cur.execute("""
        SELECT pc.collection_id FROM project_collections pc
        JOIN collections c ON c.id = pc.collection_id
        WHERE pc.project_id = %s AND c.user_id = %s
    """, (project_id, user))
    ids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return ids or None


@app.post("/ask")
def ask(req: AskRequest, user: str = Depends(current_user)):
    collection_ids = project_collection_ids(req.project_id, user) if req.project_id else None
    answer, metrics = coordinate(req.question, req.history, collection_ids=collection_ids)
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


# ------------------------------------------------------------------ collections
@app.get("/collections")
def list_collections(user: str = Depends(current_user)):
    conn = db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT c.*,
            (SELECT count(DISTINCT COALESCE(company, source_file))
             FROM chunks ch WHERE ch.collection_id = c.id) AS documents
        FROM collections c WHERE c.user_id = %s ORDER BY c.created_at DESC
    """, (user,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {"collections": [dict(r) for r in rows]}


@app.post("/collections")
def create_collection(body: CollectionCreate, user: str = Depends(current_user)):
    conn = db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("INSERT INTO collections (user_id, name) VALUES (%s, %s) RETURNING *",
                (user, body.name))
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return dict(row)


@app.delete("/collections/{cid}")
def delete_collection(cid: int, user: str = Depends(current_user)):
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM collections WHERE id = %s AND user_id = %s", (cid, user))
    if not cur.fetchone():
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Collection not found")
    cur.execute("DELETE FROM chunks WHERE collection_id = %s", (cid,))
    cur.execute("DELETE FROM collections WHERE id = %s", (cid,))  # project links cascade
    conn.commit()
    cur.close()
    conn.close()
    return {"deleted": True}


@app.get("/collections/{cid}/documents")
def collection_documents(cid: int, user: str = Depends(current_user)):
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM collections WHERE id = %s AND user_id = %s", (cid, user))
    if not cur.fetchone():
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Collection not found")
    cur.execute("""
        SELECT COALESCE(company, source_file) AS title, count(*) AS chunks
        FROM chunks WHERE collection_id = %s
        GROUP BY COALESCE(company, source_file) ORDER BY title
    """, (cid,))
    docs = [{"title": t, "chunks": c} for t, c in cur.fetchall()]
    cur.close()
    conn.close()
    return {"documents": docs}


@app.post("/collections/{cid}/upload")
async def upload_document(cid: int, file: UploadFile = File(...),
                          user: str = Depends(current_user)):
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM collections WHERE id = %s AND user_id = %s", (cid, user))
    if not cur.fetchone():
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Collection not found")

    data = await file.read()
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
        text = "\n".join(doc[i].get_text() for i in range(doc.page_count))
        doc.close()
    except Exception:
        cur.close()
        conn.close()
        raise HTTPException(status_code=400, detail="Could not read the PDF")
    if not text.strip():
        cur.close()
        conn.close()
        raise HTTPException(status_code=400, detail="No extractable text in the PDF")

    title = file.filename
    chunks = chunk_text([(1, text)])
    embeddings = embed_chunks(chunks)
    cur.execute("DELETE FROM chunks WHERE collection_id = %s AND source_file = %s", (cid, title))
    for (chunk, _p), emb in zip(chunks, embeddings):
        emb_str = "[" + ",".join(str(x) for x in emb) + "]"
        cur.execute("""
            INSERT INTO chunks (content, embedding, source_file, company,
                                doc_type, source, collection_id, user_id)
            VALUES (%s, %s, %s, %s, 'upload', 'Uploaded', %s, %s)
        """, (chunk, emb_str, title, title, cid, user))
    conn.commit()
    cur.close()
    conn.close()
    return {"title": title, "chunks": len(chunks)}


# ------------------------------------------------------------------ project <-> collections
@app.get("/projects/{pid}/collections")
def get_project_collections(pid: int, user: str = Depends(current_user)):
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM projects WHERE id = %s AND user_id = %s", (pid, user))
    if not cur.fetchone():
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Project not found")
    cur.execute("SELECT collection_id FROM project_collections WHERE project_id = %s", (pid,))
    ids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return {"collection_ids": ids}


@app.put("/projects/{pid}/collections")
def set_project_collections(pid: int, body: AttachCollections,
                            user: str = Depends(current_user)):
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM projects WHERE id = %s AND user_id = %s", (pid, user))
    if not cur.fetchone():
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Project not found")
    valid = []
    if body.collection_ids:
        cur.execute("SELECT id FROM collections WHERE id = ANY(%s) AND user_id = %s",
                    (list(body.collection_ids), user))
        valid = [r[0] for r in cur.fetchall()]
    cur.execute("DELETE FROM project_collections WHERE project_id = %s", (pid,))
    for cid in valid:
        cur.execute("INSERT INTO project_collections (project_id, collection_id) VALUES (%s, %s)",
                    (pid, cid))
    conn.commit()
    cur.close()
    conn.close()
    return {"collection_ids": valid}

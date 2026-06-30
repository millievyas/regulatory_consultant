"""FastAPI backend for the regulatory consultant.

Run:  python -m uvicorn app:app --reload
Then open http://127.0.0.1:8000

Every endpoint returns real data from the pgvector store (or the live agents).
"""

import psycopg2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agents import coordinate

app = FastAPI(title="R Path AI")


def db():
    return psycopg2.connect(dbname="regintel")


class AskRequest(BaseModel):
    question: str
    history: list = []   # [{"role": "user"|"assistant", "content": str}, ...]


@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/overview")
def overview():
    """Real corpus aggregates straight from the chunks table."""
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
    """Real per-document listing (one row per distinct document)."""
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        SELECT COALESCE(company, source_file) AS title, source, doc_type,
               count(*) AS chunks, max(url) AS url, max(issue_date) AS issue_date
        FROM chunks
        WHERE source IS NOT NULL
        GROUP BY COALESCE(company, source_file), source, doc_type
        ORDER BY source, title
    """)
    docs = [{"title": t, "source": s, "doc_type": dt,
             "chunks": c, "url": u, "issue_date": d}
            for t, s, dt, c, u, d in cur.fetchall()]
    cur.close()
    conn.close()
    return {"documents": docs}


@app.post("/ask")
def ask(req: AskRequest):
    answer, metrics = coordinate(req.question, req.history)
    return {"answer": answer, "metrics": metrics}

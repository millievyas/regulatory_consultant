"""Live retrieval — the self-healing fallback.

When the local corpus can't answer a question well, fetch from authoritative
government APIs in real time, persist the results into the corpus (so the next
user gets them instantly), and let the agent re-search over them.

Only official sources are used, so everything stays citable and current:
  - eCFR API            : exact 21 CFR section text, always the latest issue
  - openFDA enforcement : drug recalls / enforcement actions
  - openFDA drug label  : approved labeling (indications, warnings, dosage)

This module never raises out to the caller — live retrieval is best-effort. If
a source is down or returns nothing, we simply add nothing and the agent answers
from whatever the corpus already had.
"""

import re
import requests
import psycopg2

from bs4 import BeautifulSoup
from ingest import chunk_text, embed_chunks

HEADERS = {"User-Agent": "RPathAI/1.0 (regulatory research)"}

# Matches CFR citations like "21 CFR 211.84", "21 CFR 211", "21CFR211.100".
CFR_RE = re.compile(r"(\d{1,2})\s*CFR\s*(\d+)(?:\.(\d+))?", re.I)


def _store(doc, conn):
    """Chunk, embed, and upsert one fetched doc. Idempotent by URL."""
    text = doc.get("text")
    if not text:
        return 0
    chunks = chunk_text([(1, text)])
    embeddings = embed_chunks(chunks)

    cur = conn.cursor()
    cur.execute("DELETE FROM chunks WHERE url = %s", (doc["url"],))
    for (chunk, _page), embedding in zip(chunks, embeddings):
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
        cur.execute(
            """INSERT INTO chunks
               (content, embedding, source_file, company, subject, issue_date, url, source, doc_type)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (chunk, embedding_str, doc["title"], doc["title"],
             doc.get("subject", ""), doc.get("date", ""),
             doc["url"], doc["source"], doc.get("doc_type", "")),
        )
    conn.commit()
    cur.close()
    return len(chunks)


# --------------------------------------------------------------------------- #
# eCFR
# --------------------------------------------------------------------------- #
def _ecfr_latest_date(title="21"):
    r = requests.get("https://www.ecfr.gov/api/versioner/v1/titles.json",
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    for t in r.json()["titles"]:
        if str(t["number"]) == str(title):
            return t["latest_issue_date"]
    return None


def fetch_ecfr(title, part, section=None):
    """Fetch a single CFR part or section's current text from the eCFR API."""
    date = _ecfr_latest_date(title)
    if not date:
        return None
    url = (f"https://www.ecfr.gov/api/versioner/v1/full/{date}/"
           f"title-{title}.xml?part={part}")
    if section:
        url += f"&section={section}"
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    text = BeautifulSoup(r.text, "xml").get_text("\n", strip=True)
    if not text:
        return None
    label = f"{title} CFR {section or part}"
    return {"text": text, "title": f"{label} (eCFR, live)", "source": "eCFR",
            "doc_type": "regulation", "url": url, "date": date}


# --------------------------------------------------------------------------- #
# openFDA
# --------------------------------------------------------------------------- #
def fetch_openfda_enforcement(query, limit=5):
    """Fetch recent drug recalls / enforcement actions matching the query."""
    url = "https://api.fda.gov/drug/enforcement.json"
    r = requests.get(url, params={"search": query, "limit": limit},
                     headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return None
    results = r.json().get("results", [])
    if not results:
        return None
    blocks = []
    for x in results:
        blocks.append(
            f"Recalling firm: {x.get('recalling_firm', '')}\n"
            f"Product: {(x.get('product_description', '') or '')[:400]}\n"
            f"Reason: {x.get('reason_for_recall', '')}\n"
            f"Classification: {x.get('classification', '')} | "
            f"Status: {x.get('status', '')} | "
            f"Initiated: {x.get('recall_initiation_date', '')}"
        )
    return {"text": "\n\n---\n\n".join(blocks),
            "title": f"openFDA recalls/enforcement — '{query}' (live)",
            "source": "openFDA", "doc_type": "enforcement",
            "url": f"{url}?search={query}&limit={limit}", "date": ""}


def fetch_openfda_label(query, limit=3):
    """Fetch approved drug labeling matching the query."""
    url = "https://api.fda.gov/drug/label.json"
    r = requests.get(url, params={"search": query, "limit": limit},
                     headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return None
    results = r.json().get("results", [])
    if not results:
        return None

    def field(x, key):
        v = x.get(key)
        v = " ".join(v) if isinstance(v, list) else (v or "")
        return v[:600]

    blocks = []
    for x in results:
        of = x.get("openfda", {}) or {}
        name = ", ".join(of.get("brand_name") or of.get("generic_name") or ["(drug)"])
        blocks.append(
            f"Drug: {name}\n"
            f"Indications: {field(x, 'indications_and_usage')}\n"
            f"Warnings: {field(x, 'warnings')}\n"
            f"Dosage: {field(x, 'dosage_and_administration')}"
        )
    return {"text": "\n\n---\n\n".join(blocks),
            "title": f"openFDA drug label — '{query}' (live)",
            "source": "openFDA", "doc_type": "label",
            "url": f"{url}?search={query}&limit={limit}", "date": ""}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def live_retrieve_and_persist(query):
    """Fetch from the right authoritative source(s) for this query and persist.

    Routing: a CFR citation in the query -> eCFR; otherwise -> openFDA recalls +
    labels. Returns the number of documents stored (0 if nothing found or on any
    error). Best-effort: never raises.
    """
    docs = []
    try:
        m = CFR_RE.search(query)
        if m:
            title, part, sub = m.group(1), m.group(2), m.group(3)
            section = f"{part}.{sub}" if sub else None
            d = fetch_ecfr(title, part, section)
            if d:
                docs.append(d)
        else:
            for fn in (fetch_openfda_enforcement, fetch_openfda_label):
                try:
                    d = fn(query)
                    if d:
                        docs.append(d)
                except Exception:
                    pass
    except Exception as e:
        print(f"  [live] lookup failed: {e}")
        return 0

    if not docs:
        return 0

    try:
        conn = psycopg2.connect(dbname="regintel")
        stored = sum(1 for d in docs if _store(d, conn))
        conn.close()
    except Exception as e:
        print(f"  [live] persist failed: {e}")
        return 0

    print(f"  [live] fetched + stored {stored} doc(s) for: {query!r}")
    return stored

# Regulatory Intelligence — Multi-Agent RAG System

An automated regulatory knowledge base that ingests guidance, regulations, and
enforcement actions from multiple health authorities and answers complex compliance
questions using specialized, tool-using AI agents coordinated by an LLM router.

Built on Python, PostgreSQL/pgvector, and OpenAI. Documents are retrieved
programmatically (APIs and scrapers) rather than downloaded by hand, and the
ingestion layer is a pluggable adapter architecture so new sources are easy to add.

---

## What it does

Ask a regulatory question (e.g. *"Compare how FDA and EMA approach process validation"*
or *"What CGMP violations were found, and at which companies?"*) and the system:

1. **Routes** the question to the relevant specialist agent(s) via an LLM coordinator.
2. Each agent **searches the knowledge base as a tool** — it decides what to search
   for, can scope by authority or document type, and can search multiple times.
3. It **synthesizes a grounded, cited answer** using only the retrieved evidence, and
   declines to answer when the evidence isn't present (no hallucinated facts).

---

## Knowledge base

A single `pgvector` store (~30K chunks) spanning multiple authorities and document types:

| Authority | Region | Document types |
| --- | --- | --- |
| eCFR (Title 21) | US | Regulations (the law) |
| FDA | US | Guidance documents, Warning letters (enforcement) |
| EMA | EU | Scientific guidelines |
| MHRA | UK | Guidance documents |

This gives the "regulatory trifecta" for the US — what the rule is (CFR), how it's
interpreted (guidance), and what happens when it's broken (warning letters) — plus
cross-jurisdiction coverage for EU/UK.

---

## Results

Measured on a hand-labeled 30-question routing benchmark and a 10-query performance
sample (OpenAI `gpt-4o-mini` + `text-embedding-3-small`).

**Coordinator: keyword vs. LLM routing**

| Metric | Keyword router | LLM router | Change |
| --- | --- | --- | --- |
| Routing accuracy | 73% | **93%** | +20 pts |
| Agents fired / query | 1.30 | 1.10 | −15% |
| Cost / query | $0.00039 | $0.00033 | −15% |

Replacing keyword routing with an LLM coordinator improved accuracy **and** lowered
cost, because the smarter router stopped firing redundant agents.

**Agents: fixed retrieval vs. tool use**

| Metric | Fixed pipeline | Tool-using agents |
| --- | --- | --- |
| Latency / query | ~5.3s | ~22s |
| Cost / query | $0.00033 | $0.0014 |

Tool use buys multi-step, filter-aware reasoning (e.g. searching FDA and EMA
separately to compare them) at a real latency/cost premium — a deliberate tradeoff.

---

## Architecture

```
                        User question
                              |
                              v
                   ┌──────────────────────┐
                   │   Coordinator (LLM)   │   route() — picks specialist agent(s)
                   └──────────┬───────────┘
              ┌───────────────┼───────────────┐
              v               v               v
       ┌────────────┐  ┌────────────┐  ┌──────────────┐
       │ Regulatory │  │  Quality   │  │ Manufacturing │   tool-using agents
       └─────┬──────┘  └─────┬──────┘  └──────┬───────┘
             └───────────────┼────────────────┘
                             v
                   ┌──────────────────────┐
                   │  search_documents()   │   filtered semantic search (a tool)
                   └──────────┬───────────┘
                              v
                   ┌──────────────────────┐
                   │ PostgreSQL + pgvector │  chunks + embeddings + metadata
                   └──────────────────────┘
                              ^
                              │  ingestion (adapter architecture)
        ┌──────────────┬──────┴───────┬──────────────┐
   fda_adapter    ecfr_adapter   pdf_adapter     web_adapter
   (warning        (Title 21      (guidance       (PDF or HTML
    letters)        XML API)       PDFs)           guidance)
```

Each adapter `yield`s documents in one normalized shape, so the storage/retrieval/agent
layers are completely source-agnostic. Adding a source = writing one adapter.

---

## Tech stack

- **Language:** Python
- **LLM + embeddings:** OpenAI (`gpt-4o-mini`, `text-embedding-3-small`, 1536-dim)
- **Vector store:** PostgreSQL with the `pgvector` extension
- **Ingestion:** `requests` + `beautifulsoup4` (crawl/parse), `trafilatura` (HTML text
  extraction), `pymupdf` (in-memory PDF text extraction), the eCFR public API

---

## Project structure

| File | Purpose |
| --- | --- |
| `scrape.py` | Ingestion: source adapters (FDA, eCFR, PDF, web), normalization, and storage. |
| `ingest.py` | Shared helpers: `chunk_text`, `embed_chunks`. |
| `query.py` | Retrieval: `embed_query` and `search` (filtered top-K over pgvector). |
| `agents.py` | Specialist agents, the LLM coordinator (`route`), the `search_documents` tool, and the tool-using agent loop. |
| `eval_routing.py` | Routing-accuracy benchmark (30 labeled questions). |
| `eval_perf.py` | Latency / token / cost benchmark over a batch of queries. |
| `test.py` | Quick smoke test for DB connectivity + search. |

---

## Setup

### 1. PostgreSQL + pgvector

```sql
CREATE DATABASE regintel;
\c regintel
CREATE EXTENSION vector;

CREATE TABLE chunks (
    id          SERIAL PRIMARY KEY,
    content     TEXT NOT NULL,
    embedding   VECTOR(1536),
    source_file TEXT,
    page        INTEGER,
    company     TEXT,
    subject     TEXT,
    issue_date  TEXT,
    url         TEXT,
    source      TEXT,        -- authority: FDA, eCFR, EMA, MHRA
    doc_type    TEXT         -- regulation, guidance, warning_letter
);
```

### 2. Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
```

---

## Usage

Ingest the corpus (edit the source list at the bottom of `scrape.py`; the full eCFR
Title 21 load is large and slow, so comment it out if you only want guidance/letters):

```bash
python scrape.py
```

Ask questions interactively (routes → tool-using agents → cited answer):

```bash
python agents.py
```

Run the benchmarks:

```bash
python eval_routing.py     # routing accuracy
python eval_perf.py        # latency / tokens / cost
```

---

## Roadmap

- **Memory** — conversation history for follow-up questions; per-project investigation state.
- **More tools** — `summarize_doc`, section-level citations.
- **More authorities** — additional ICH markets (some sites need a browser-grade fetcher
  such as `curl_cffi` to get past CDN bot protection).
- **Evaluation** — expand the benchmark to ~100 questions; add retrieval and citation metrics.
- **Performance** — run routed agents concurrently to cut multi-agent latency.
- **Deployment** — S3 for originals, RDS (pgvector), ECS; swap OpenAI for AWS Bedrock.

---

## Notes / limitations

- Benchmarks are small (30 routing questions, 10 perf queries) — directional for a
  portfolio project, not production-grade statistics.
- Routing ground-truth labels reflect a self-defined scope for each agent's domain.
- Scrapers depend on each site's current structure and may need maintenance.
- Some authority sites (e.g. Health Canada, TGA) block plain `requests` via CDN
  fingerprinting and aren't currently ingested.

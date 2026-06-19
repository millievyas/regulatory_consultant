# Regulatory Intelligence — Multi-Agent RAG System

A regulatory intelligence system that automatically ingests FDA warning letters and
answers complex compliance questions using multiple specialized AI agents coordinated
by an LLM-based router.

Built on Python, PostgreSQL/pgvector, and OpenAI. Designed to scale from a handful of
documents to a large corpus, with an automated scraper so documents are retrieved
programmatically rather than downloaded by hand.

---

## What it does

Ask a regulatory question (e.g. *"What CGMP manufacturing violations were found, and at
which companies?"*) and the system:

1. Routes the question to the relevant specialist agent(s) using an LLM coordinator.
2. Retrieves the most relevant document chunks from a vector store (semantic search).
3. Generates a grounded, cited answer using only the retrieved evidence.

Answers cite the company they came from, and the system declines to answer when the
evidence isn't present (no hallucinated facts).

---

## Results

Measured on a hand-labeled 30-question routing benchmark and a 10-query performance
sample (OpenAI `gpt-4o-mini` + `text-embedding-3-small`).

| Metric | Keyword router | LLM router | Change |
| --- | --- | --- | --- |
| Routing accuracy | 73% | **93%** | +20 pts |
| Agents fired / query | 1.30 | 1.10 | −15% |
| Tokens / query | 1,577 | 1,337 | −15% |
| Cost / query | $0.00039 | $0.00033 | −15% |
| Latency / query | 5.71s | 5.28s | −8% |

Key finding: replacing keyword routing with an LLM coordinator improved routing accuracy
**and** reduced cost/latency, because the smarter router stopped firing redundant agents —
the savings from fewer (expensive) agent calls outweighed the one (cheap) router call.

---

## Architecture

```
                        User question
                              |
                              v
                   ┌──────────────────────┐
                   │   Coordinator (LLM)   │   route() — picks agent(s)
                   └──────────┬───────────┘
              ┌───────────────┼───────────────┐
              v               v               v
       ┌────────────┐  ┌────────────┐  ┌──────────────┐
       │ Regulatory │  │  Quality   │  │ Manufacturing │   specialist agents
       │   Agent    │  │   Agent    │  │    Agent      │
       └─────┬──────┘  └─────┬──────┘  └──────┬───────┘
             └───────────────┼────────────────┘
                             v
                   ┌──────────────────────┐
                   │  RAG retrieval layer  │   search() over pgvector
                   └──────────┬───────────┘
                              v
                   ┌──────────────────────┐
                   │  PostgreSQL + pgvector │  chunks + embeddings + metadata
                   └──────────────────────┘
                              ^
                              │  ingestion
                   ┌──────────────────────┐
                   │   FDA scraper (HTML)  │  crawl listing → fetch → chunk → embed
                   └──────────────────────┘
```

Each "agent" is the same retrieve-then-answer flow with a specialized system prompt.
The coordinator is an LLM that reads the question and returns which agents should run.

---

## Tech stack

- **Language:** Python
- **LLM + embeddings:** OpenAI (`gpt-4o-mini`, `text-embedding-3-small`, 1536-dim)
- **Vector store:** PostgreSQL with the `pgvector` extension
- **Ingestion:** `requests` + `beautifulsoup4` (crawl) and `trafilatura` (clean text extraction); optional PDF path via `pymupdf`

---

## Project structure

| File | Purpose |
| --- | --- |
| `scrape.py` | Crawl the FDA warning-letter listing, fetch each letter, and ingest it (fetch → chunk → embed → store with metadata). |
| `ingest.py` | Shared ingestion helpers (`chunk_text`, `embed_chunks`); optional PDF ingestion functions. |
| `query.py` | Retrieval layer — `embed_query` and `search` (semantic top-K over pgvector). |
| `agents.py` | Specialist agents, the LLM coordinator (`route`), and `coordinate` (the interactive multi-agent app). |
| `eval_routing.py` | Routing-accuracy benchmark (30 labeled questions). |
| `eval_perf.py` | Latency / token / cost benchmark over a batch of queries. |

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
    url         TEXT
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

Ingest recent FDA warning letters (adjust `max_pages` to control how many):

```bash
python scrape.py
```

Ask questions interactively:

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

- **Phase 4** — tool use + memory: explicit tools (`search_documents`, `retrieve_guidance`) and per-project investigation state.
- **Phase 5** — full evaluation suite: expand the benchmark to ~100 questions; add retrieval-accuracy and citation/grounding metrics.
- **Phase 6** — AWS deployment: S3 for documents, RDS (PostgreSQL + pgvector), ECS Fargate for the backend; swap OpenAI for AWS Bedrock.
- **Optimization** — concurrent agent execution to reduce multi-agent latency.

---

## Notes / limitations

- Benchmark (30 questions) and performance sample (10 queries) are small; numbers are
  directional for a portfolio project, not production-grade statistics.
- Routing ground-truth labels reflect a self-defined scope for each agent's domain.
- The scraper depends on FDA's current page structure and may need maintenance if the
  site changes.

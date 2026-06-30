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

Evaluated with three independent harnesses (OpenAI `gpt-4o-mini` as generator/router,
`gpt-4o` as the grounding judge).

**Routing — keyword vs. LLM coordinator** (75-question benchmark, same test for both)

| Router | Accuracy |
| --- | --- |
| Keyword baseline | 66.7% |
| LLM coordinator | **88.0%** (+21 pts) |

The LLM router's biggest gains were on semantically-phrased questions keyword matching
missed (quality category 30% → 100%). Remaining errors concentrate at the
manufacturing/quality domain boundary — largely defensible overlaps. A targeted router
refinement was tested and reverted after it regressed overall accuracy.

**Retrieval accuracy** (22 questions, unfiltered search over the whole corpus)

| Metric | Result |
| --- | --- |
| Source hit@5 | **100%** |
| Source hit@3 | 95% |
| Source hit@1 | 73% |

The correct authority's evidence reaches the top-5 every time, despite a ~10:1 corpus
imbalance toward regulations.

**Grounding** (12 questions incl. negative controls, judged by `gpt-4o`)

100% of answers fully supported by retrieved context, 0 hallucinations — every
negative-control question (revenues, fines, headcount) correctly refused.

**Performance**

- The LLM coordinator also cut cost ~15% vs. keyword routing by firing fewer redundant agents.
- Running routed agents **concurrently** reduced multi-agent query latency **55%**
  (24.2s → 11.0s, ~2.2× speedup) at no extra token cost.
- Tool-using agents trade higher latency/cost for multi-step, filter-aware reasoning
  (e.g. searching FDA and EMA separately to compare them) — a deliberate tradeoff.

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
| `agents.py` | Specialist agents, the LLM coordinator (`route`), the tool suite (`search_documents`, `list_documents`, `fetch_document`), the tool-using agent loop, conversation memory, and concurrent agent execution. |
| `eval_routing.py` | Routing-accuracy benchmark (75 questions; keyword vs. LLM). |
| `eval_retrieval.py` | Retrieval-accuracy benchmark (source hit@k). |
| `eval_grounding.py` | Grounding / hallucination benchmark (LLM-as-judge). |
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
python eval_routing.py     # routing accuracy (keyword vs. LLM)
python eval_retrieval.py   # retrieval accuracy (source hit@k)
python eval_grounding.py   # grounding / hallucination (LLM-as-judge)
python eval_perf.py        # latency / tokens / cost
```

---

## Roadmap

Done: multi-authority ingestion, LLM coordinator, tool-using agents (search / list /
fetch), conversation memory, concurrent agent execution, and a three-axis eval suite
(routing, retrieval, grounding).

Next:

- **UI** — a web front end over the coordinator.
- **More authorities** — additional ICH markets (some sites need a browser-grade fetcher
  such as `curl_cffi` to get past CDN bot protection).
- **Per-project memory** — persistent investigation state across sessions.
- **Section-level citations** — cite the specific regulation/section, not the whole document.
- **Deployment** — S3 for originals, RDS (pgvector), ECS; swap OpenAI for AWS Bedrock.

---

## Notes / limitations

- Benchmarks are modest in size (75 routing, 22 retrieval, 12 grounding) — directional
  for a portfolio project, not production-grade statistics.
- Routing ground-truth labels reflect a self-defined scope for each agent's domain.
- Scrapers depend on each site's current structure and may need maintenance.
- Some authority sites (e.g. Health Canada, TGA) block plain `requests` via CDN
  fingerprinting and aren't currently ingested.

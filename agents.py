import time
import json

from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from query import search, list_documents, fetch_document

# OpenAI pricing per token — VERIFY current rates at openai.com/pricing
PRICE_IN  = 0.15 / 1_000_000   # gpt-4o-mini input tokens
PRICE_OUT = 0.60 / 1_000_000   # gpt-4o-mini output tokens

load_dotenv()
client = OpenAI()

# Each agent is just a name + a specialized system prompt
GROUNDING = (
    " Cite the source and document for every claim (the company name for warning "
    "letters; the authority and document title for guidance and regulations). "
    "Use ONLY the provided context; if the answer isn't there, say so. Never "
    "invent facts, citations, or URLs."
)

AGENTS = {
    "regulatory": (
        "You are a regulatory affairs specialist covering multiple authorities "
        "(FDA, EMA, MHRA) and US regulations (FD&C Act, 21 CFR). Focus on "
        "regulatory pathways, statutory and regulatory requirements, guidance "
        "interpretation, misbranding, enforcement actions, and compliance risk."
        + GROUNDING
    ),
    "quality": (
        "You are a pharmaceutical quality systems specialist working across "
        "authorities (FDA, EMA, MHRA). Focus on CAPA, deviations, SOPs, quality "
        "unit responsibilities, data integrity, audits, and stability and method "
        "validation."
        + GROUNDING
    ),
    "manufacturing": (
        "You are a pharmaceutical manufacturing specialist working across "
        "authorities (FDA, EMA, MHRA). Focus on CGMP, process validation, "
        "sterility, batch records, equipment, components, and contamination "
        "control."
        + GROUNDING
    ),
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Search the regulatory knowledge base for relevant text. "
                       "Call multiple times with different queries/filters if needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "query":    {"type": "string", "description": "What to search for"},
                "source":   {"type": "string", "enum": ["FDA", "eCFR", "EMA", "MHRA"],
                             "description": "Optional: restrict to one authority"},
                "doc_type": {"type": "string", "enum": ["regulation", "guidance", "warning_letter"],
                             "description": "Optional: restrict to one document type"},
            },
            "required": ["query"],
        },
    },
}

LIST_TOOL = {
    "type": "function",
    "function": {
        "name": "list_documents",
        "description": "List documents in the knowledge base (title, source, type). "
                       "Use to discover what exists, e.g. which companies have warning letters.",
        "parameters": {
            "type": "object",
            "properties": {
                "source":   {"type": "string", "enum": ["FDA", "eCFR", "EMA", "MHRA"]},
                "doc_type": {"type": "string", "enum": ["regulation", "guidance", "warning_letter"]},
            },
        },
    },
}

FETCH_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_document",
        "description": "Fetch the full text of ONE document by its exact title "
                       "(as shown by list_documents) to read or summarize the whole thing.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Exact document title"},
            },
            "required": ["title"],
        },
    },
}

def execute_search(args):
    results = search(args["query"], top_k=5,
                     source=args.get("source"), doc_type=args.get("doc_type"))
    if not results:
        return "No matching documents found."
    return "\n\n".join(
        f"[{source} | {company}]\n{content}"
        for content, company, subject, url, source in results
    )

def execute_list(args):
    rows = list_documents(source=args.get("source"), doc_type=args.get("doc_type"))
    if not rows:
        return "No documents found."
    return "\n".join(f"- {title}  [{source}/{doc_type}]" for title, source, doc_type in rows)

def execute_fetch(args):
    text = fetch_document(args["title"])
    return text or f"No document found with title '{args['title']}'."

# Map tool name -> executor, and the list of tool specs offered to the model.
TOOL_DISPATCH = {
    "search_documents": execute_search,
    "list_documents":   execute_list,
    "fetch_document":   execute_fetch,
}
ALL_TOOLS = [SEARCH_TOOL, LIST_TOOL, FETCH_TOOL]

def route_keyword(query):
    """Simple keyword-based routing baseline. Returns a list of agent names."""
    q = query.lower()
    selected = []

    if any(word in q for word in
           ["fda", "pathway", "ind", "nda", "bla", "compliance",
            "regulation", "regulatory", "statute", "misbrand", "violation"]):
        selected.append("regulatory")

    if any(word in q for word in
           ["capa", "deviation", "sop", "quality", "audit", "stability"]):
        selected.append("quality")

    if any(word in q for word in
           ["manufactur", "cgmp", "process", "steril", "batch",
            "contamination", "validation", "equipment", "component"]):
        selected.append("manufacturing")

    if not selected:                 # default to regulatory
        selected = ["regulatory"]

    return selected

def route(query, history=None):
    system_prompt = (
        "You are a router for a regulatory analysis system. Decide which "
        "specialist agents should answer the user's question.\n\n"
        "Available agents:\n"
        "- regulatory: FDA pathways (IND/NDA/BLA), FD&C Act statutes, misbranding, "
        "enforcement actions, import alerts, legal/compliance risk.\n"
        "- quality: CAPA, deviations, SOPs, quality unit duties, data integrity, "
        "falsified records, audits, stability programs, analytical method validation.\n"
        "- manufacturing: CGMP, process validation, sterility, contamination, "
        "equipment, batch records, component/raw-material testing.\n\n"
        "Return ONLY a JSON array of the relevant agent names, e.g. [\"quality\"] "
        "or [\"regulatory\",\"manufacturing\"]. Include an agent only if genuinely "
        "relevant; prefer the smallest correct set."
    )

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages += history
    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()

    try:
        names = json.loads(raw)
    except json.JSONDecodeError:
        names = [a for a in AGENTS if a in raw.lower()]   # fallback

    selected = [n for n in names if n in AGENTS]
    if not selected:
        selected = ["regulatory"]
    return selected

# ---------------------------------------------------------------------------
# Agents + coordinator
# ---------------------------------------------------------------------------

def run_agent_tools(agent_name, query, history=None, max_steps=5):
    messages = [
        {"role": "system", "content": AGENTS[agent_name] +
            " You have three tools: search_documents (semantic search, optionally "
            "filtered by source/doc_type), list_documents (discover what documents "
            "exist), and fetch_document (read a whole document by title). Gather "
            "evidence with these before answering; you may call them several times."},
    ]

    if history:
        messages += history
    messages.append({"role": "user", "content": query})

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=ALL_TOOLS,
        )
        u = response.usage
        usage["prompt_tokens"]     += u.prompt_tokens
        usage["completion_tokens"] += u.completion_tokens
        usage["cost"]              += u.prompt_tokens * PRICE_IN + u.completion_tokens * PRICE_OUT

        msg = response.choices[0].message
        if not msg.tool_calls:
            return msg.content, usage

        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            fn = TOOL_DISPATCH.get(tc.function.name)
            print(f"  [tool] {tc.function.name}({args})")
            result = fn(args) if fn else f"Unknown tool: {tc.function.name}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "Stopped after max search steps.", usage

def coordinate(query, history=None):
    history = history or []
    start = time.perf_counter()
    agents = route(query, history)

    # Run the routed agents concurrently. They're independent and I/O-bound
    # (waiting on the API + DB), so threads overlap those waits instead of
    # running the agents back-to-back.
    with ThreadPoolExecutor(max_workers=len(agents)) as pool:
        outputs = list(pool.map(
            lambda name: (name, run_agent_tools(name, query, history)),
            agents
        ))

    metrics = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "agents": len(agents)}
    sections = []
    for name, (answer, usage) in outputs:
        metrics["prompt_tokens"]     += usage["prompt_tokens"]
        metrics["completion_tokens"] += usage["completion_tokens"]
        metrics["cost"]              += usage["cost"]
        sections.append(f"### {name.title()} Agent\n{answer}")

    metrics["latency"] = time.perf_counter() - start
    print(f"\n[Metrics] agents={metrics['agents']} latency={metrics['latency']:.2f}s "
          f"tokens={metrics['prompt_tokens']}in+{metrics['completion_tokens']}out "
          f"cost=${metrics['cost']:.5f}")
    return "\n\n".join(sections), metrics

if __name__ == "__main__":
    print("Multi-agent regulatory consultant. Type 'quit' to exit.")
    history = []
    while True:
        query = input("\nQuestion: ")
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query.strip():
            continue

        text, _ = coordinate(query, history)
        print("\n" + text)

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": text})
        history = history[-6:]   # keep the last 3 turns
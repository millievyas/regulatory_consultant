from query import search
from openai import OpenAI
from dotenv import load_dotenv
import time
import json

# OpenAI pricing per token — VERIFY current rates at openai.com/pricing
PRICE_IN  = 0.15 / 1_000_000   # gpt-4o-mini input tokens
PRICE_OUT = 0.60 / 1_000_000   # gpt-4o-mini output tokens

load_dotenv()
client = OpenAI();

# Each agent is just a name + a specialized system prompt
AGENTS = {
    "regulatory": (
        "You are an FDA Regulatory Affairs specialist. Focus on regulatory "
        "pathways (IND/NDA/BLA), FDA guidance, statutory violations, and "
        "compliance risk. Cite the company for each claim. "
        "Use ONLY the provided context; if it's not there, say so."
    ),
    "quality": (
        "You are a pharmaceutical Quality Systems specialist. Focus on CAPA, "
        "deviations, SOP compliance, quality unit responsibilities, and audit "
        "findings. Cite the company for each claim. "
        "Use ONLY the provided context; if it's not there, say so."
    ),
    "manufacturing": (
        "You are a pharmaceutical Manufacturing specialist. Focus on CGMP, "
        "process validation, sterility, batch records, equipment, and "
        "contamination risks. Cite the company for each claim. "
        "Use ONLY the provided context; if it's not there, say so."
    ),
}

def run_agent(agent_name, query, top_k=5):
    results = search(query, top_k)
    context = "\n\n".join(
        f"[{source} | {company}]\n{content}"
        for content, company, subject, url, source in results
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": AGENTS[agent_name]},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    u = response.usage
    usage = {
        "prompt_tokens":     u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "cost": u.prompt_tokens * PRICE_IN + u.completion_tokens * PRICE_OUT,
    }
    return response.choices[0].message.content, usage

def route_keyword(query):
    """Simple keyword-based routing. Returns a list of agent names to run."""
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

def route(query):
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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
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

def coordinate(query):
    start = time.perf_counter()
    agents = route(query)

    metrics = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "agents": len(agents)}
    sections = []
    for name in agents:
        answer, usage = run_agent(name, query)
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
    while True:
        query = input("\nQuestion: ")
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query.strip():
            continue
        text, _ = coordinate(query)      # unpack: text + metrics
        print("\n" + text)
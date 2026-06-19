from query import search
from openai import OpenAI
from dotenv import load_dotenv

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
        f"[Company: {company} | Subject: {subject}]\n{content}"
        for content, company, subject, url in results
    )

    system_prompt = AGENTS[agent_name]
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content

def route(query):
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

def coordinate(query):
    agents = route(query)
    print(f"[Coordinator] routing to: {', '.join(agents)}\n")

    sections = []
    for name in agents:
        answer = run_agent(name, query)
        sections.append(f"### {name.title()} Agent\n{answer}")

    return "\n\n".join(sections)

if __name__ == "__main__":
    print("Multi-agent regulatory consultant. Type 'quit' to exit.")
    while True:
        query = input("\nQuestion: ")
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query.strip():
            continue
        print("\n" + coordinate(query))
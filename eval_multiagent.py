from agents import coordinate

# Questions deliberately spanning multiple domains, so the router picks
# several specialist agents — which is what agent-level concurrency speeds up.
MULTI_AGENT_QUERIES = [
    "What are the regulatory, quality, and manufacturing failures at the cited companies?",
    "Compare regulatory compliance and quality-system expectations for data integrity.",
    "What manufacturing deviations and CGMP violations led to regulatory consequences?",
    "Summarize the legal, quality-system, and CGMP issues found in the warning letters.",
    "How do process validation requirements and their compliance implications compare across FDA and EMA?",
    "What deviations, sterility concerns, and statutory violations were identified?",
]

def run_batch(queries):
    rows = []
    for i, q in enumerate(queries, 1):
        _, m = coordinate(q)
        rows.append(m)
        print(f"[{i}] agents={m['agents']}  latency={m['latency']:.2f}s")

    n = len(queries)
    print(f"\n=== Multi-agent set: {n} queries ===")
    print(f"avg agents/query: {sum(m['agents']  for m in rows)/n:.2f}")
    print(f"avg latency:      {sum(m['latency'] for m in rows)/n:.2f}s")
    print(f"avg cost/query:   ${sum(m['cost']   for m in rows)/n:.5f}")

if __name__ == "__main__":
    run_batch(MULTI_AGENT_QUERIES)
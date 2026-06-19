from agents import coordinate

QUERIES = [
    "What CGMP manufacturing violations were found?",
    "Were any records falsified or backdated?",
    "What misbranding claims were made?",
    "Describe process validation concerns.",
    "What are the compliance and quality system failures?",
    "Which companies were warned and why?",
    "Were incoming components tested for identity?",
    "What deviations and CAPA issues were cited?",
    "Summarize the manufacturing and quality problems.",
    "What legal consequences follow if violations aren't fixed?",
]

def run_batch(queries):
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "latency": 0.0, "agents": 0}
    for q in queries:
        _, m = coordinate(q)
        for k in totals:
            totals[k] += m[k]

    n = len(queries)
    tot_tokens = totals["prompt_tokens"] + totals["completion_tokens"]
    print(f"\n=== Averages over {n} queries ===")
    print(f"avg latency:      {totals['latency']/n:.2f}s")
    print(f"avg agents/query: {totals['agents']/n:.2f}")
    print(f"avg tokens/query: {tot_tokens/n:.0f}")
    print(f"avg cost/query:   ${totals['cost']/n:.5f}")
    print(f"total tokens:     {tot_tokens}")
    print(f"total cost:       ${totals['cost']:.4f}")

if __name__ == "__main__":
    run_batch(QUERIES)
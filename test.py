"""Quick smoke test: confirms the DB is reachable and search returns results."""

from query import search

results = search("data integrity expectations", top_k=3)
print(f"search returned {len(results)} result(s)")
for content, company, subject, url, source in results:
    print(f"  [{source}] {company[:60]}")

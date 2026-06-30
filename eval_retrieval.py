"""Retrieval-accuracy eval.

For each question we label the authority (source) whose documents SHOULD answer
it, run unfiltered semantic search over the whole corpus, and check whether a
chunk from that source appears in the top-K. Unfiltered is the honest test:
when an agent filters by source, retrieval is correct by construction.
"""

from collections import defaultdict
from query import search

# (question, expected_source) — the authority whose documents should answer it.
RETRIEVAL_BENCHMARK = [
    # eCFR (US regulations)
    ("What does 21 CFR 211.84 require for testing of incoming components?", "eCFR"),
    ("What are the CGMP requirements for finished pharmaceuticals?",        "eCFR"),
    ("What does 21 CFR 211.22 say about the quality control unit?",         "eCFR"),
    ("What does 21 CFR 211.100 require for production and process controls?", "eCFR"),
    ("What are the regulatory requirements for stability testing of drug products?", "eCFR"),
    ("What does Title 21 require for control of drug components and containers?", "eCFR"),

    # FDA (guidance + warning letters)
    ("What are the three stages of process validation in FDA guidance?",    "FDA"),
    ("What does FDA guidance say about data integrity and CGMP compliance?", "FDA"),
    ("What does FDA recommend for aseptic processing of sterile drug products?", "FDA"),
    ("How does FDA say to investigate out-of-specification (OOS) results?",  "FDA"),
    ("What CGMP violations did Sante Manufacturing receive in its warning letter?", "FDA"),
    ("Which companies received telehealth misbranding warning letters?",    "FDA"),
    ("What does FDA guidance say about quality agreements in contract manufacturing?", "FDA"),

    # EMA (EU guidelines)
    ("What does EMA require for sterilisation of the medicinal product?",    "EMA"),
    ("What is EMA's guidance on process validation for finished products?",  "EMA"),
    ("What does EMA say about health-based exposure limits in shared facilities?", "EMA"),
    ("What does EMA require for the quality of water for pharmaceutical use?", "EMA"),
    ("What is EMA's guidance on stability testing of existing active substances?", "EMA"),

    # MHRA (UK guidance)
    ("What is MHRA's guidance on GxP data integrity?",                       "MHRA"),
    ("What does MHRA say about good manufacturing and distribution practice?", "MHRA"),
    ("What is the MHRA UK guideline on decentralised manufacture?",          "MHRA"),
    ("What does MHRA expect for GMP pre-inspection compliance reporting?",   "MHRA"),
]


def first_hit_rank(question, expected, top_k=5):
    results = search(question, top_k=top_k)   # unfiltered: whole corpus
    sources = [source for (_content, _company, _subject, _url, source) in results]
    for i, s in enumerate(sources, 1):
        if s == expected:
            return i, sources
    return None, sources


def evaluate(top_k=5):
    n = len(RETRIEVAL_BENCHMARK)
    hit1 = hit3 = hit5 = 0
    by_source = defaultdict(lambda: [0, 0])   # source -> [hit@5, total]
    misses = []

    for question, expected in RETRIEVAL_BENCHMARK:
        rank, sources = first_hit_rank(question, expected, top_k)
        by_source[expected][1] += 1
        if rank is not None:
            hit5 += 1
            by_source[expected][0] += 1
            hit1 += rank <= 1
            hit3 += rank <= 3
        else:
            misses.append((question, expected, sources))

    print(f"Retrieval accuracy (n={n}):")
    print(f"  hit@1: {hit1}/{n} = {hit1 / n * 100:.0f}%")
    print(f"  hit@3: {hit3}/{n} = {hit3 / n * 100:.0f}%")
    print(f"  hit@5: {hit5}/{n} = {hit5 / n * 100:.0f}%")

    print("\nhit@5 by expected source:")
    for src in sorted(by_source):
        h, t = by_source[src]
        print(f"  {src:6s} {h}/{t}  ({h / t * 100:.0f}%)")

    if misses:
        print(f"\nMisses ({len(misses)}):")
        for q, exp, srcs in misses:
            print(f"  [{exp}] {q}")
            print(f"     got: {srcs}")


if __name__ == "__main__":
    evaluate()

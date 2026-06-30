"""Grounding eval (LLM-as-judge).

For each question: retrieve context, generate an answer constrained to that
context, then have a judge model decide whether the answer is supported by the
context. Negative-control questions (not answerable from the corpus) should be
refused — a correct refusal counts as 'supported', so this measures both
faithfulness and hallucination resistance.

Note: this tests the retrieve-then-answer grounding step against a FIXED context
(cleaner to judge than the multi-search agent loop). For a more independent
judge, set JUDGE_MODEL to a stronger model than the generator.
"""

import json
from openai import OpenAI
from dotenv import load_dotenv
from query import search

load_dotenv()
client = OpenAI()

GEN_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o"   # consider a stronger model for a more independent judge

GROUNDING_QUESTIONS = [
    # --- Answerable from the corpus ---
    "What does 21 CFR 211.84 require for testing of incoming components?",
    "What are the three stages of FDA process validation?",
    "What does EMA require for sterilisation of medicinal products?",
    "What CGMP violations did Sante Manufacturing receive in its warning letter?",
    "What is MHRA's guidance on GxP data integrity?",
    "How does FDA say to investigate out-of-specification (OOS) results?",
    "What are the CGMP responsibilities of the quality control unit?",
    "What does EMA say about health-based exposure limits in shared facilities?",

    # --- Negative controls (NOT in the corpus — the system should refuse) ---
    "What were Sante Manufacturing's annual revenues?",
    "How many employees does Ready Med have?",
    "What monetary fines were imposed on the cited companies?",
    "Who is the CEO of the warned firms?",
]

GEN_SYSTEM = (
    "You are a regulatory analyst. Answer using ONLY the provided context. "
    "If the answer is not in the context, say you cannot find it. "
    "Do not invent facts, citations, or URLs."
)

JUDGE_SYSTEM = (
    "You are a strict grader checking whether an ANSWER is grounded in the given "
    "CONTEXT.\n"
    "- 'supported' = every factual claim is backed by the context, OR the answer "
    "correctly states the information cannot be found.\n"
    "- 'partial' = mostly grounded but includes at least one claim not in the context.\n"
    "- 'unsupported' = key claims are absent from the context (hallucination).\n"
    'Respond with ONLY a JSON object: {"verdict": "supported|partial|unsupported", '
    '"reason": "<one short sentence>"}.'
)


def answer_from_context(question, top_k=5):
    results = search(question, top_k=top_k)
    context = "\n\n".join(
        f"[{source} | {company}]\n{content}"
        for content, company, subject, url, source in results
    )
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content, context


def judge(question, answer, context):
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content":
                f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:\n{answer}"},
        ],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        low = raw.lower()
        verdict = "unsupported" if "unsupported" in low else (
                  "partial" if "partial" in low else "supported")
        return {"verdict": verdict, "reason": raw[:80]}


def evaluate():
    counts = {"supported": 0, "partial": 0, "unsupported": 0}
    for q in GROUNDING_QUESTIONS:
        answer, context = answer_from_context(q)
        result = judge(q, answer, context)
        v = result.get("verdict", "partial")
        counts[v] = counts.get(v, 0) + 1
        print(f"[{v:11s}] {q}")
        print(f"     {result.get('reason', '')}")

    n = len(GROUNDING_QUESTIONS)
    print(f"\nGrounding verdicts (n={n}): "
          f"supported={counts['supported']}, partial={counts['partial']}, "
          f"unsupported={counts['unsupported']}")
    print(f"Fully-grounded rate: {counts['supported'] / n * 100:.0f}%")


if __name__ == "__main__":
    evaluate()

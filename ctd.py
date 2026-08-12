"""CTD submission builder — Module 3 (Quality) + Module 2.3 (QOS).

Given a project's document collections (the company's own validation reports,
technical reports, SOPs, batch records, etc.), draft each CTD Quality section
grounded ONLY in those source documents, cite every source, and flag sections
with no supporting data as gaps.

IMPORTANT: the output is a first draft for qualified regulatory review — never a
submission-ready package. It never fabricates data, and it never draws Module 3
content from the shared regulatory corpus (only the company's own documents).
"""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from dotenv import load_dotenv
from query import search

load_dotenv()
client = OpenAI()

DRAFT_MODEL = "gpt-4o-mini"

# CTD Module 3.2 Quality section ontology (per ICH M4Q, the CTD-Quality guideline).
# Each section: id, title, the requirement it must satisfy, and search hints used to
# pull the right source documents from the company's collection.
CTD_SECTIONS = [
    # ---- 3.2.S Drug Substance ----
    {"id": "3.2.S.1", "title": "General Information (Drug Substance)",
     "requirement": "Nomenclature, structure, and general properties of the drug substance.",
     "hints": ["drug substance nomenclature structure molecular formula",
               "drug substance physicochemical properties"]},
    {"id": "3.2.S.2", "title": "Manufacture (Drug Substance)",
     "requirement": "Manufacturer(s), description of the manufacturing process and process controls, "
                    "control of materials, controls of critical steps and intermediates, process "
                    "validation/evaluation, and manufacturing process development.",
     "hints": ["drug substance manufacturing process synthesis route",
               "drug substance process validation critical process parameters"]},
    {"id": "3.2.S.3", "title": "Characterisation (Drug Substance)",
     "requirement": "Elucidation of structure and other characteristics; impurities.",
     "hints": ["drug substance structure elucidation characterisation",
               "drug substance impurities related substances"]},
    {"id": "3.2.S.4", "title": "Control of Drug Substance",
     "requirement": "Specification, analytical procedures, validation of analytical procedures, batch "
                    "analyses, and justification of specification.",
     "hints": ["drug substance specification acceptance criteria",
               "drug substance analytical method validation batch analysis"]},
    {"id": "3.2.S.5", "title": "Reference Standards or Materials (Drug Substance)",
     "requirement": "Reference standards or materials used for testing of the drug substance.",
     "hints": ["reference standard qualification drug substance"]},
    {"id": "3.2.S.6", "title": "Container Closure System (Drug Substance)",
     "requirement": "Description and specifications of the container closure system for the drug substance.",
     "hints": ["drug substance container closure packaging system"]},
    {"id": "3.2.S.7", "title": "Stability (Drug Substance)",
     "requirement": "Stability summary and conclusions, post-approval stability protocol and commitment, "
                    "and stability data.",
     "hints": ["drug substance stability study data shelf life retest",
               "drug substance stability protocol storage conditions"]},
    # ---- 3.2.P Drug Product ----
    {"id": "3.2.P.1", "title": "Description and Composition of the Drug Product",
     "requirement": "Description of the dosage form and its composition.",
     "hints": ["drug product composition formulation dosage form"]},
    {"id": "3.2.P.2", "title": "Pharmaceutical Development",
     "requirement": "Development of the formulation, overages, physicochemical and biological properties, "
                    "manufacturing process development, container closure system, microbiological "
                    "attributes, and compatibility.",
     "hints": ["pharmaceutical development formulation development",
               "quality by design design of experiments critical quality attributes"]},
    {"id": "3.2.P.3", "title": "Manufacture (Drug Product)",
     "requirement": "Manufacturer(s), batch formula, description of the manufacturing process and process "
                    "controls, controls of critical steps and intermediates, and process validation/evaluation.",
     "hints": ["drug product manufacturing process batch formula",
               "drug product process validation batch record critical steps"]},
    {"id": "3.2.P.4", "title": "Control of Excipients",
     "requirement": "Specifications, analytical procedures, and justification of specifications for excipients; "
                    "excipients of human/animal origin; novel excipients.",
     "hints": ["excipient specification control analytical procedure"]},
    {"id": "3.2.P.5", "title": "Control of Drug Product",
     "requirement": "Specification(s), analytical procedures, validation of analytical procedures, batch "
                    "analyses, characterisation of impurities, and justification of specification(s).",
     "hints": ["drug product specification acceptance criteria release",
               "finished product analytical method validation impurities batch analysis"]},
    {"id": "3.2.P.6", "title": "Reference Standards or Materials (Drug Product)",
     "requirement": "Reference standards or materials used for the drug product.",
     "hints": ["reference standard drug product"]},
    {"id": "3.2.P.7", "title": "Container Closure System (Drug Product)",
     "requirement": "Container closure system, including materials of construction and suitability "
                    "(protection, compatibility, safety, performance).",
     "hints": ["drug product container closure system suitability",
               "packaging materials of construction extractables leachables"]},
    {"id": "3.2.P.8", "title": "Stability (Drug Product)",
     "requirement": "Stability summary and conclusions, post-approval stability protocol and commitment, "
                    "and stability data.",
     "hints": ["drug product stability study shelf life storage conditions",
               "finished product stability data protocol"]},
]

# Sources that belong to the shared global corpus — NEVER used as Module 3 content.
GLOBAL_SOURCES = {"FDA", "eCFR", "EMA", "MHRA", "ICH", "EU GMP", "WHO", "openFDA"}

SECTION_SYSTEM = (
    "You are a CMC regulatory writer drafting one section of CTD Module 3 (Quality) "
    "per ICH M4Q. Draft ONLY from the provided source documents (the company's own "
    "validation reports, technical reports, SOPs, batch records, etc.). Write in the "
    "formal, factual style of a regulatory submission. Cite each source document by "
    "its title inline, e.g. (Source: <title>). If the sources do not contain enough "
    "to satisfy the section requirement, do NOT invent content — instead begin your "
    "response with 'GAP:' and state precisely what data is missing. Never fabricate "
    "results, batch numbers, specifications, or analytical values."
)

QOS_SYSTEM = (
    "You are a CMC regulatory writer preparing the CTD Module 2.3 Quality Overall "
    "Summary (QOS) per ICH M4Q. Summarise the Module 3 section drafts provided into a "
    "concise, well-structured QOS covering the drug substance (2.3.S) and drug product "
    "(2.3.P). Use ONLY the provided drafts; do not introduce new facts or data."
)


def _draft_section(section, collection_ids):
    """Retrieve the company's own source docs for one CTD section and draft it."""
    hits, seen = [], set()
    for hint in section["hints"]:
        for content, company, subject, url, src in search(
                hint, top_k=5, collection_ids=collection_ids, collections_only=True):
            if src in GLOBAL_SOURCES:            # defensive: company docs only
                continue
            key = (company, content[:80])
            if key in seen:
                continue
            seen.add(key)
            hits.append({"content": content, "company": company, "url": url})

    if not hits:
        return {"id": section["id"], "title": section["title"], "status": "gap",
                "draft": (f"No source documents found for {section['id']} "
                          f"({section['title']}). Required: {section['requirement']}"),
                "sources": []}

    context = "\n\n".join(f"[{h['company']}]\n{h['content']}" for h in hits)
    sources = sorted({h["company"] for h in hits})
    user = (f"CTD Section {section['id']} — {section['title']}\n"
            f"Section requirement: {section['requirement']}\n\n"
            f"Source documents (the company's own):\n{context}\n\n"
            f"Draft this CTD section from the sources above.")

    resp = client.chat.completions.create(
        model=DRAFT_MODEL, temperature=0,
        messages=[{"role": "system", "content": SECTION_SYSTEM},
                  {"role": "user", "content": user}])
    draft = resp.choices[0].message.content.strip()
    status = "partial" if draft.upper().startswith("GAP:") else "drafted"
    return {"id": section["id"], "title": section["title"], "status": status,
            "draft": draft, "sources": sources}


def _build_qos(sections):
    """Summarise the drafted Module 3 sections into the 2.3 Quality Overall Summary."""
    drafted = [s for s in sections if s["status"] != "gap"]
    if not drafted:
        return "Insufficient source data to draft a Quality Overall Summary."
    body = "\n\n".join(f"{s['id']} {s['title']}:\n{s['draft'][:1500]}" for s in drafted)
    resp = client.chat.completions.create(
        model=DRAFT_MODEL, temperature=0,
        messages=[{"role": "system", "content": QOS_SYSTEM},
                  {"role": "user", "content": body}])
    return resp.choices[0].message.content.strip()


def build_module3(collection_ids, max_workers=5):
    """Draft CTD Module 3 (Quality) + Module 2.3 (QOS) from a project's collections.

    Returns a structured result: per-section drafts with status + citations, the QOS,
    and a coverage summary. Sections are drafted concurrently (I/O-bound on the API).
    """
    if not collection_ids:
        raise ValueError("CTD Module 3 requires the project to have at least one "
                         "document collection attached.")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        sections = list(pool.map(lambda s: _draft_section(s, collection_ids),
                                 CTD_SECTIONS))

    qos = _build_qos(sections)
    counts = Counter(s["status"] for s in sections)
    coverage = {"drafted": counts.get("drafted", 0),
                "partial": counts.get("partial", 0),
                "gap":     counts.get("gap", 0),
                "total":   len(sections)}
    return {"module": "CTD Module 3 (Quality) + 2.3 QOS",
            "disclaimer": "AI-generated first draft grounded in the company's own "
                          "documents. Requires qualified regulatory/CMC review before "
                          "any use in a submission.",
            "coverage": coverage, "sections": sections, "qos": qos}


if __name__ == "__main__":
    # Quick manual test against a collection id passed on the command line.
    import sys
    ids = [int(x) for x in sys.argv[1:]] or None
    result = build_module3(ids)
    print(json.dumps(result["coverage"], indent=2))
    for s in result["sections"]:
        print(f"[{s['status']:8s}] {s['id']} {s['title']}  (sources: {', '.join(s['sources']) or '—'})")

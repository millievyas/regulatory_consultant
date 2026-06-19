from agents import route

# (question, expected agents) — your hand-labeled ground truth
BENCHMARK = [
    # --- Regulatory (pathways, statutes, misbranding, enforcement) ---
    ("What is the FDA regulatory pathway involved?",                  {"regulatory"}),
    ("What statutory violations under the FD&C Act were cited?",      {"regulatory"}),
    ("What misbranding or false and misleading claims were made?",    {"regulatory"}),
    ("What import alerts or enforcement actions were taken?",         {"regulatory"}),
    ("Do the 503A or 503B exemptions apply to these compounds?",      {"regulatory"}),
    ("What are the legal consequences if violations aren't fixed?",   {"regulatory"}),
    ("Were any unapproved new drugs being marketed?",                 {"regulatory"}),
    ("Why were the products deemed adulterated under the statute?",   {"regulatory"}),

    # --- Quality (CAPA, deviations, SOPs, QU, data integrity, audits) ---
    ("Were any records falsified or backdated?",                      {"quality"}),
    ("What CAPA issues were cited?",                                  {"quality"}),
    ("Did the quality unit fail its responsibilities?",              {"quality"}),
    ("What deviations from standard operating procedures occurred?",  {"quality"}),
    ("What audit or inspection findings about the quality system?",   {"quality"}),
    ("Were stability testing programs adequate?",                     {"quality"}),
    ("Were analytical methods properly validated?",                   {"quality"}),
    ("What data integrity problems were found?",                      {"quality"}),

    # --- Manufacturing (CGMP, process validation, sterility, equipment) ---
    ("What CGMP manufacturing violations were found?",                {"manufacturing"}),
    ("Describe the process validation and qualification concerns.",   {"manufacturing"}),
    ("Were there sterility or microbial contamination risks?",        {"manufacturing"}),
    ("Were incoming components tested for identity?",                 {"manufacturing"}),
    ("Were batch records or production controls deficient?",          {"manufacturing"}),
    ("Was manufacturing equipment properly maintained?",             {"manufacturing"}),
    ("Were there cross-contamination risks in production?",           {"manufacturing"}),
    ("Was the water system qualified and monitored?",                {"manufacturing"}),

    # --- Multi-agent (genuinely span domains) ---
    ("What are the compliance and quality system failures?",          {"regulatory", "quality"}),
    ("Summarize manufacturing and quality problems in the letters.",  {"manufacturing", "quality"}),
    ("How do manufacturing failures create legal/compliance risk?",   {"regulatory", "manufacturing"}),
    ("What deviations and CGMP process violations occurred?",         {"quality", "manufacturing"}),
    ("Give a full regulatory, quality, and manufacturing risk review.", {"regulatory", "quality", "manufacturing"}),
    ("What systemic quality issues drove the regulatory violations?", {"regulatory", "quality"}),
]

def evaluate():
    correct = 0
    for question, expected in BENCHMARK:
        predicted = set(route(question))
        ok = predicted == expected
        correct += ok
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {question}")
        if not ok:
            print(f"        expected {expected}, got {predicted}")
    pct = correct / len(BENCHMARK) * 100
    print(f"\nRouting accuracy: {correct}/{len(BENCHMARK)} = {pct:.0f}%")

if __name__ == "__main__":
    evaluate()
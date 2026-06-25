from collections import defaultdict
from agents import route, route_keyword

# (question, expected agents) — hand-labeled ground truth.
# Labels reflect IDEAL semantic routing (what a perfect coordinator should pick),
# not what any particular router currently does. Agents are DOMAIN specialists,
# so "compare FDA and EMA sterility" is manufacturing (one domain, two authorities).
# Labels are judgment calls on domain boundaries — adjust to your own definitions.
BENCHMARK = [
    # --- Regulatory ---
    ("What is the FDA regulatory pathway involved?",                    {"regulatory"}),
    ("What statutory violations under the FD&C Act were cited?",        {"regulatory"}),
    ("What misbranding or false and misleading claims were made?",      {"regulatory"}),
    ("What import alerts or enforcement actions were taken?",           {"regulatory"}),
    ("Do the 503A or 503B exemptions apply to these compounds?",        {"regulatory"}),
    ("What are the legal consequences if violations aren't fixed?",     {"regulatory"}),
    ("Were any unapproved new drugs being marketed?",                   {"regulatory"}),
    ("Why were the products deemed adulterated under the statute?",     {"regulatory"}),
    ("Which sections of the FD&C Act were violated?",                   {"regulatory"}),
    ("What labeling violations were identified?",                       {"regulatory"}),
    ("How does EMA's marketing authorization process work?",            {"regulatory"}),
    ("What compliance risks does the company face?",                    {"regulatory"}),
    ("Were the promotional claims false or misleading?",                {"regulatory"}),
    ("Was the drug misbranded under section 502?",                      {"regulatory"}),
    ("What import restrictions were imposed?",                          {"regulatory"}),
    ("What FDA enforcement action was taken against the company?",      {"regulatory"}),
    ("What regulatory submission requirements apply for approval?",     {"regulatory"}),
    ("Is this product an unapproved new drug requiring an NDA?",        {"regulatory"}),
    ("What telehealth marketing violations were cited?",                {"regulatory"}),
    ("What legal basis supports the warning letter?",                   {"regulatory"}),

    # --- Quality ---
    ("Were any records falsified or backdated?",                        {"quality"}),
    ("What CAPA issues were cited?",                                    {"quality"}),
    ("Did the quality unit fail its responsibilities?",                {"quality"}),
    ("What deviations from standard operating procedures occurred?",    {"quality"}),
    ("What audit or inspection findings about the quality system?",     {"quality"}),
    ("Were stability testing programs adequate?",                       {"quality"}),
    ("Were analytical methods properly validated?",                     {"quality"}),
    ("What data integrity problems were found?",                        {"quality"}),
    ("Were corrective and preventive actions adequate?",                {"quality"}),
    ("Did the firm investigate out-of-specification results?",          {"quality"}),
    ("Were change controls properly managed?",                          {"quality"}),
    ("What documentation and record-keeping problems existed?",         {"quality"}),
    ("Was data altered or deleted?",                                    {"quality"}),
    ("What ALCOA data integrity principles were violated?",             {"quality"}),
    ("Did the quality unit approve the batch correctly?",               {"quality"}),
    ("Were SOPs followed during laboratory testing?",                   {"quality"}),
    ("What internal audit deficiencies were noted?",                    {"quality"}),
    ("Compare FDA and MHRA data integrity expectations.",               {"quality"}),
    ("Were complaints and recalls handled appropriately?",              {"quality"}),
    ("Was the stability program scientifically sound?",                 {"quality"}),

    # --- Manufacturing ---
    ("What CGMP manufacturing violations were found?",                  {"manufacturing"}),
    ("Describe the process validation and qualification concerns.",     {"manufacturing"}),
    ("Were there sterility or microbial contamination risks?",          {"manufacturing"}),
    ("Were incoming components tested for identity?",                   {"manufacturing"}),
    ("Were batch records or production controls deficient?",            {"manufacturing"}),
    ("Was manufacturing equipment properly maintained?",               {"manufacturing"}),
    ("Were there cross-contamination risks in production?",             {"manufacturing"}),
    ("Was the water system qualified and monitored?",                  {"manufacturing"}),
    ("Was the aseptic processing environment adequate?",                {"manufacturing"}),
    ("Were cleaning validation procedures sufficient?",                 {"manufacturing"}),
    ("Was the manufacturing process properly qualified?",               {"manufacturing"}),
    ("Were raw materials tested before use in production?",             {"manufacturing"}),
    ("What contamination control measures were lacking?",               {"manufacturing"}),
    ("Was the facility designed appropriately for sterile manufacturing?", {"manufacturing"}),
    ("Were environmental monitoring results acceptable?",               {"manufacturing"}),
    ("What equipment qualification gaps existed?",                      {"manufacturing"}),
    ("Were APIs manufactured under proper controls?",                   {"manufacturing"}),
    ("Was process performance qualification completed?",                {"manufacturing"}),
    ("Compare FDA and EMA sterility and aseptic processing requirements.", {"manufacturing"}),
    ("Were finished dosage forms made to specification?",               {"manufacturing"}),

    # --- Multi-agent (genuinely span domains) ---
    ("What are the compliance and quality system failures?",            {"regulatory", "quality"}),
    ("Summarize manufacturing and quality problems in the letters.",    {"manufacturing", "quality"}),
    ("How do manufacturing failures create legal/compliance risk?",     {"regulatory", "manufacturing"}),
    ("What deviations and CGMP process violations occurred?",           {"quality", "manufacturing"}),
    ("Give a full regulatory, quality, and manufacturing risk review.", {"regulatory", "quality", "manufacturing"}),
    ("What systemic quality issues drove the regulatory violations?",   {"regulatory", "quality"}),
    ("Summarize the manufacturing defects and resulting enforcement.",  {"manufacturing", "regulatory"}),
    ("What process-validation and quality-control problems were cited?", {"manufacturing", "quality"}),
    ("How did manufacturing and data-integrity failures lead to legal consequences?",
                                                                        {"regulatory", "quality", "manufacturing"}),
    ("What CGMP violations and statutory breaches occurred?",           {"manufacturing", "regulatory"}),
    ("Assess the deviations, contamination risks, and compliance exposure.",
                                                                        {"regulatory", "quality", "manufacturing"}),
    ("What quality and manufacturing root causes underlie the violations?", {"quality", "manufacturing"}),
    ("Compare the compliance and quality-system expectations across FDA and EMA.",
                                                                        {"regulatory", "quality"}),
    ("What regulatory and manufacturing issues were found at the cited firms?",
                                                                        {"regulatory", "manufacturing"}),
    ("Give a combined quality and regulatory assessment of the firm.",  {"regulatory", "quality"}),
]


def evaluate(router=route, label="LLM router", show_failures=True):
    correct = 0
    fails = []
    # accuracy per "category" = sorted tuple of expected agents
    by_cat = defaultdict(lambda: [0, 0])   # cat -> [correct, total]

    for question, expected in BENCHMARK:
        predicted = set(router(question))
        ok = predicted == expected
        correct += ok
        cat = "+".join(sorted(expected))
        by_cat[cat][0] += ok
        by_cat[cat][1] += 1
        if not ok:
            fails.append((question, expected, predicted))

    n = len(BENCHMARK)
    print(f"{label} accuracy: {correct}/{n} = {correct / n * 100:.1f}%\n")

    print("By category (expected agents):")
    for cat in sorted(by_cat):
        c, t = by_cat[cat]
        print(f"  {cat:40s} {c}/{t}  ({c / t * 100:.0f}%)")

    if show_failures and fails:
        print(f"\nFailures ({len(fails)}):")
        for q, exp, pred in fails:
            print(f"  expected {sorted(exp)}, got {sorted(pred)}")
            print(f"    {q}")
    return correct / n * 100


if __name__ == "__main__":
    print("=== Keyword router (baseline) ===")
    kw = evaluate(route_keyword, "Keyword router", show_failures=False)
    print("\n=== LLM router ===")
    llm = evaluate(route, "LLM router")
    print(f"\nImprovement: {kw:.1f}% -> {llm:.1f}%  (+{llm - kw:.1f} pts) "
          f"on the same {len(BENCHMARK)}-question benchmark")

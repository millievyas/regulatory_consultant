"""Warning-letter metadata — the structured / analytics layer.

Complements the semantic (RAG) layer. The `warning_letters` table holds one row
per FDA warning letter (company, office, subject, issue_date, year, url), so the
agents can answer QUANTITATIVE questions — counts, date ranges, by-office, trends,
top companies — that similarity search fundamentally can't.

sync_metadata() populates it from the same FDA DataTables feed the scraper uses,
metadata only (no bodies, ~36 fast requests). Everything else is read-only query.
"""

import datetime
import psycopg2

# Office substrings that identify drug + biologics/vaccine enforcement — the same
# set used to scope the body corpus, so counts line up with what we ingested.
DRUG_BIOLOGICS = ("drug", "biologic", "pharmaceutical quality", "manufacturing quality")


def _parse_date(s):
    s = (s or "").strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def _office_clause(scope):
    """Return (sql, params) restricting by issuing office.

    scope='drug_biologics' (default) -> drug + biologics/vaccine offices.
    scope='all'                      -> every FDA warning letter.
    """
    if scope == "all":
        return "", []
    clause = "(" + " OR ".join(["LOWER(office) LIKE %s"] * len(DRUG_BIOLOGICS)) + ")"
    return clause, [f"%{k}%" for k in DRUG_BIOLOGICS]


def _build_where(scope="drug_biologics", year=None, year_from=None, year_to=None,
                 office_contains=None, subject_contains=None, company_contains=None):
    conds, params = [], []
    oclause, oparams = _office_clause(scope)
    if oclause:
        conds.append(oclause); params += oparams
    if year is not None:
        conds.append("year = %s"); params.append(year)
    if year_from is not None:
        conds.append("year >= %s"); params.append(year_from)
    if year_to is not None:
        conds.append("year <= %s"); params.append(year_to)
    if office_contains:
        conds.append("LOWER(office) LIKE %s"); params.append(f"%{office_contains.lower()}%")
    if subject_contains:
        conds.append("LOWER(subject) LIKE %s"); params.append(f"%{subject_contains.lower()}%")
    if company_contains:
        conds.append("LOWER(company) LIKE %s"); params.append(f"%{company_contains.lower()}%")
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    return where, params


def _run(sql, params):
    conn = psycopg2.connect(dbname="regintel")
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close(); conn.close()
    return rows


# --------------------------------------------------------------------------- #
# Read-only queries
# --------------------------------------------------------------------------- #
def count_letters(**filters):
    where, params = _build_where(**filters)
    return _run(f"SELECT COUNT(*) FROM warning_letters {where}", params)[0][0]


def list_letters(limit=25, **filters):
    where, params = _build_where(**filters)
    rows = _run(f"""SELECT company, office, issue_date, subject, url
                    FROM warning_letters {where}
                    ORDER BY issue_date DESC NULLS LAST
                    LIMIT %s""", params + [limit])
    return rows


def letters_by_year(**filters):
    where, params = _build_where(**filters)
    return _run(f"""SELECT year, COUNT(*) FROM warning_letters {where}
                    {'AND' if where else 'WHERE'} year IS NOT NULL
                    GROUP BY year ORDER BY year""", params)


def top_companies(limit=10, **filters):
    where, params = _build_where(**filters)
    return _run(f"""SELECT company, COUNT(*) c FROM warning_letters {where}
                    GROUP BY company ORDER BY c DESC LIMIT %s""", params + [limit])


# --------------------------------------------------------------------------- #
# Populate / refresh (metadata only)
# --------------------------------------------------------------------------- #
def sync_metadata():
    """Populate/refresh warning_letters from the FDA DataTables feed. Idempotent
    by URL. Metadata only — no bodies, no embeddings — so it's fast."""
    from scrape import fda_letter_list   # lazy import (heavy deps)
    conn = psycopg2.connect(dbname="regintel")
    cur = conn.cursor()
    n = 0
    for l in fda_letter_list():
        issue = _parse_date(l.get("issue_date"))
        cur.execute("""
            INSERT INTO warning_letters (company, office, subject, issue_date, year, url)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO UPDATE SET
                company    = EXCLUDED.company,
                office     = EXCLUDED.office,
                subject    = EXCLUDED.subject,
                issue_date = EXCLUDED.issue_date,
                year       = EXCLUDED.year
        """, (l["company"], l["office"], l["subject"], issue,
              issue.year if issue else None, l["url"]))
        n += 1
        if n % 500 == 0:
            conn.commit(); print(f"  {n} letters synced…")
    conn.commit(); cur.close(); conn.close()
    print(f"  done — {n} warning letters in the metadata table")
    return n


if __name__ == "__main__":
    sync_metadata()

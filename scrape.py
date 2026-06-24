import time
import requests
import trafilatura
import psycopg2
import pymupdf
import socket
import urllib3.util.connection as urllib3_cn

from bs4 import BeautifulSoup
from urllib.parse import urljoin
from ingest import chunk_text, embed_chunks

def _allowed_gai_family():
    return socket.AF_INET   # force IPv4 only

urllib3_cn.allowed_gai_family = _allowed_gai_family

BASE = "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/compliance-actions-and-activities/warning-letters"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

FDA_GUIDANCE_DOCS = [
    {"url": "https://www.fda.gov/files/drugs/published/Process-Validation--General-Principles-and-Practices.pdf",
     "title": "FDA Guidance - Process Validation: General Principles and Practices"},
    {"url": "https://www.fda.gov/media/119267/download",
     "title": "FDA Guidance - Data Integrity and Compliance With Drug CGMP: Q&A"},
    {"url": "https://www.fda.gov/media/71026/download",
     "title": "FDA Guidance - Sterile Drug Products Produced by Aseptic Processing (CGMP)"},
    {"url": "https://www.fda.gov/media/90425/download",
     "title": "FDA Guidance - CGMP for Phase 1 Investigational Drugs"},
    {"url": "https://www.fda.gov/files/drugs/published/Q7-Good-Manufacturing-Practice-Guidance-for-Active-Pharmaceutical-Ingredients-Guidance-for-Industry.pdf",
     "title": "FDA Guidance - Q7 GMP for Active Pharmaceutical Ingredients"},
    {"url": "https://www.fda.gov/files/drugs/published/Analytical-Procedures-and-Methods-Validation-for-Drugs-and-Biologics.pdf",
     "title": "FDA Guidance - Analytical Procedures and Methods Validation for Drugs and Biologics"},
    {"url": "https://www.fda.gov/files/drugs/published/Bioanalytical-Method-Validation-Guidance-for-Industry.pdf",
     "title": "FDA Guidance - Bioanalytical Method Validation"},
    {"url": "https://www.fda.gov/media/158416/download",
     "title": "FDA Guidance - Investigating Out-of-Specification (OOS) Test Results"},
    {"url": "https://www.fda.gov/media/86193/download",
     "title": "FDA Guidance - Contract Manufacturing Arrangements for Drugs: Quality Agreements"},
    {"url": "https://www.fda.gov/media/161201/download",
     "title": "FDA Guidance - Q2(R2) Validation of Analytical Procedures"},
    {"url": "https://www.fda.gov/media/77391/download",
     "title": "FDA Guidance - Pharmaceutical CGMPs for the 21st Century: A Risk-Based Approach"},
    {"url": "https://www.fda.gov/downloads/Drugs/Guidances/UCM070337.pdf",
     "title": "FDA Guidance - Quality Systems Approach to Pharmaceutical CGMP Regulations"},
]

EMA_DOCS = [
    {"url": "https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-process-validation-finished-products-information-data-be-provided-regulatory-submissions_en.pdf",
     "title": "EMA Guideline - Process Validation for Finished Products"},
    {"url": "https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-process-validation-manufacture-biotechnology-derived-active-substances-and-data-be-provided-regulatory-submission_en.pdf",
     "title": "EMA Guideline - Process Validation for Biotechnology-Derived Active Substances"},
    {"url": "https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-manufacture-finished-dosage-form-revision-1_en.pdf",
     "title": "EMA Guideline - Manufacture of the Finished Dosage Form (Rev 1)"},
    {"url": "https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-sterilisation-medicinal-product-active-substance-excipient-and-primary-container_en.pdf",
     "title": "EMA Guideline - Sterilisation of the Medicinal Product, Active Substance, Excipient and Primary Container"},
    {"url": "https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-setting-health-based-exposure-limits-use-risk-identification-manufacture-different-medicinal-products-shared-facilities_en.pdf",
     "title": "EMA Guideline - Setting Health-Based Exposure Limits (Shared Facilities / Cross-Contamination)"},
    {"url": "https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-quality-water-pharmaceutical-use_en.pdf",
     "title": "EMA Guideline - Quality of Water for Pharmaceutical Use"},
    {"url": "https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-stability-testing-stability-testing-existing-active-substances-and-related-finished-products_en.pdf",
     "title": "EMA Guideline - Stability Testing of Existing Active Substances and Related Finished Products"},
]

UK_DOCS = [
    {"url": "https://assets.publishing.service.gov.uk/media/5aa2b9ede5274a3e391e37f3/MHRA_GxP_data_integrity_guide_March_edited_Final.pdf",
     "title": "MHRA - GxP Data Integrity Guidance and Definitions (Rev 1)"},
    {"url": "https://assets.publishing.service.gov.uk/media/5b19553f40f0b634b1266c70/GMP_Compliance_Report_Guidelines_V_7.pdf",
     "title": "MHRA - GMP Pre-Inspection Compliance Report Guidelines"},
    {"url": "https://www.gov.uk/guidance/good-manufacturing-practice-and-good-distribution-practice",
     "title": "MHRA - Good Manufacturing Practice and Good Distribution Practice"},
    {"url": "https://www.gov.uk/guidance/decentralised-manufacture-uk-guideline-on-good-manufacturing-practice-gmp",
     "title": "MHRA - Decentralised Manufacture: UK Guideline on GMP"},
]

def get_letters_on_page(page=0):
    url = f"{BASE}?page={page}"
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status() # crash loudly if the request fails
    soup = BeautifulSoup(response.text, "html.parser")

    letters = []
    for row in soup.select("table tbody tr"):
        cells = row.find_all("td")

        if len(cells) < 5: # skip malformed rows
            continue;

        link = cells[2].find("a") # company name is column 3

        if not link:
            continue

        letters.append({
            "posted_date": cells[0].get_text(strip=True),
            "issue_date":  cells[1].get_text(strip=True),
            "company":     link.get_text(strip=True),
            "url":         urljoin("https://www.fda.gov", link["href"]),
            "office":      cells[3].get_text(strip=True),
            "subject":     cells[4].get_text(strip=True),
        })
    return letters

def get_letters(max_pages=2, delay=1.0):
    all_letters = []

    for page in range(max_pages):
        letters = get_letters_on_page(page)
        print(f"Page {page}: found {len(letters)} letters")
        all_letters.extend(letters)
        time.sleep(delay)
    return all_letters

def fetch_letter_text(url):
    html = requests.get(url, headers=HEADERS, timeout=30).text
    text = trafilatura.extract(html)      # pulls out just the main content
    return text

def ingest_letter(letter, conn):
    text = fetch_letter_text(letter["url"])
    if not text:
        print("  skipped (no text extracted)")
        return

    chunks = chunk_text([(1, text)])     # reuse your chunker; the whole letter is one "page"
    embeddings = embed_chunks(chunks)

    cur = conn.cursor()
    cur.execute("DELETE FROM chunks WHERE url = %s", (letter["url"],))   # avoid duplicates on re-run

    for (chunk, _page), embedding in zip(chunks, embeddings):
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
        cur.execute(
            """INSERT INTO chunks (content, embedding, source_file, company, subject, issue_date, url)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (chunk, embedding_str, letter["company"], letter["company"],
             letter["subject"], letter["issue_date"], letter["url"])
        )
    conn.commit()
    print(f"  stored {len(chunks)} chunks")

def ingest_recent_letters(max_pages=1, delay=1.0):
    letters = get_letters(max_pages=max_pages, delay=delay)
    conn = psycopg2.connect(dbname="regintel")
    for i, letter in enumerate(letters, 1):
        print(f"[{i}/{len(letters)}] {letter['company']}")
        try:
            ingest_letter(letter, conn)
        except Exception as e:
            print("  error:", e)
        time.sleep(delay)
    conn.close()

def ingest_document(doc, conn):
    text = doc.get("text")
    if not text:
        print("  skipped (no text extracted)")
        return

    chunks = chunk_text([(1, text)])
    embeddings = embed_chunks(chunks)

    cur = conn.cursor()
    cur.execute("DELETE FROM chunks WHERE url = %s", (doc["url"],))   # idempotent

    for (chunk, _page), embedding in zip(chunks, embeddings):
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
        cur.execute(
            """INSERT INTO chunks
               (content, embedding, source_file, company, subject, issue_date, url, source, doc_type)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (chunk, embedding_str,
             doc["title"], doc["title"],         # source_file + company both = title
             doc.get("subject", ""), doc.get("date", ""),
             doc["url"], doc["source"], doc.get("doc_type", ""))
        )
    conn.commit()
    print(f"  stored {len(chunks)} chunks")

def fda_adapter(max_pages=1, delay=1.0):
    letters = get_letters(max_pages=max_pages, delay=delay)
    for letter in letters:
        text = fetch_letter_text(letter["url"])
        yield {
            "text":     text,
            "title":    letter["company"],
            "source":   "FDA",
            "doc_type": "warning_letter",
            "subject":  letter["subject"],
            "date":     letter["issue_date"],
            "url":      letter["url"],
        }
        time.sleep(delay)

def ingest_source(adapter, conn):
    for i, doc in enumerate(adapter, 1):
        print(f"[{i}] {doc['source']}: {doc['title']}")
        try:
            ingest_document(doc, conn)
        except Exception as e:
            print("  error:", e)

def ecfr_part_numbers(date, title="21"):
    url = f"https://www.ecfr.gov/api/versioner/v1/structure/{date}/title-{title}.json"
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    parts = []
    def walk(node):
        if node.get("type") == "part" and not node.get("reserved"):
            parts.append(node["identifier"])
        for child in node.get("children", []):
            walk(child)
    walk(resp.json())
    return parts

def ecfr_latest_date(title="21"):
    resp = requests.get("https://www.ecfr.gov/api/versioner/v1/titles.json",
                        headers=HEADERS, timeout=30)
    resp.raise_for_status()
    for t in resp.json()["titles"]:
        if str(t["number"]) == str(title):
            return t["latest_issue_date"]
    raise ValueError(f"Title {title} not found")

def ecfr_adapter(parts=None, title="21", delay=0.5):
    date = ecfr_latest_date(title)
    if parts is None:
        parts = ecfr_part_numbers(date, title)
        print(f"  Title {title}: {len(parts)} parts to ingest")

    for part in parts:
        api_url = (f"https://www.ecfr.gov/api/versioner/v1/full/"
                   f"{date}/title-{title}.xml?part={part}")
        try:
            resp = requests.get(api_url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            print(f"  skipped part {part}: {e}")
            continue

        soup = BeautifulSoup(resp.text, "xml")
        text = soup.get_text(separator="\n", strip=True)
        if not text:
            continue   # skip empty parts

        yield {
            "text":     text,
            "title":    f"{title} CFR Part {part}",
            "source":   "eCFR",
            "doc_type": "regulation",
            "subject":  "",
            "date":     date,
            "url":      f"https://www.ecfr.gov/current/title-{title}/part-{part}",
        }
        time.sleep(delay)

def fetch_pdf_text(url):
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    doc = pymupdf.open(stream=resp.content, filetype="pdf")   # open from memory, not disk
    text = "\n".join(doc[i].get_text() for i in range(doc.page_count))
    doc.close()
    return text

def pdf_adapter(source, docs, delay=1.0):
    for d in docs:
        try:
            text = fetch_pdf_text(d["url"])
        except Exception as e:
            print(f"  skipped {d['title']}: {e}")
            continue
        yield {
            "text":     text,
            "title":    d["title"],
            "source":   source,
            "doc_type": d.get("doc_type", "guidance"),
            "subject":  "",
            "date":     d.get("date", ""),
            "url":      d["url"],
        }
        time.sleep(delay)

def fetch_html_text(url):
    html = requests.get(url, headers=HEADERS, timeout=60).text
    return trafilatura.extract(html)

def web_adapter(source, docs, delay=1.0):
    for d in docs:
        url = d["url"]
        try:
            text = fetch_pdf_text(url) if url.lower().endswith(".pdf") else fetch_html_text(url)
        except Exception as e:
            print(f"  skipped {d['title']}: {e}")
            continue
        if not text:
            print(f"  skipped {d['title']}: no text")
            continue
        yield {
            "text": text, "title": d["title"], "source": source,
            "doc_type": d.get("doc_type", "guidance"), "subject": "",
            "date": d.get("date", ""), "url": url,
        }
        time.sleep(delay)

if __name__ == "__main__":
    conn = psycopg2.connect(dbname="regintel")
    ingest_source(web_adapter("TGA", AUSTRALIA_DOCS), conn)
    conn.close()
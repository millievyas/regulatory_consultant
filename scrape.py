import time
import requests
import trafilatura
import psycopg2

from bs4 import BeautifulSoup
from urllib.parse import urljoin
from ingest import chunk_text, embed_chunks

BASE = "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/compliance-actions-and-activities/warning-letters"
HEADERS = {"User-Agent": "RegIntel research project (educational use)"}

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

if __name__ == "__main__":
    conn = psycopg2.connect(dbname="regintel")
    ingest_source(fda_adapter(max_pages=1), conn)
    conn.close()
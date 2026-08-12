"""Unified storage connector — SCAFFOLD.

Ingest a company's documents from wherever they live (Google Drive, SharePoint,
OneDrive, Box, Dropbox, S3, ...) through ONE vendor-agnostic interface — a
"unified file storage API" such as Apideck or Unified.to. The client authorises
their own storage account once; we stream each file's bytes server-side into the
SAME pipeline used for manual uploads (extract -> chunk -> embed -> store, scoped
per company) and never keep the raw file.

Everything vendor-specific is isolated behind UnifiedStorageClient. Pick a vendor,
implement its two methods, and the entire pipeline below already works unchanged.
"""

import os
import psycopg2
import pymupdf

from dotenv import load_dotenv
from ingest import chunk_text, embed_chunks

load_dotenv()


# --------------------------------------------------------------------------- #
# Vendor-agnostic client — THE ONLY PART THAT CHANGES PER VENDOR.
# --------------------------------------------------------------------------- #
class UnifiedStorageClient:
    """Thin wrapper over a unified file-storage API (Apideck / Unified.to / ...).

    A `connection` is one client's authorised storage account (their Drive,
    SharePoint site, Box, etc.), represented by a `token` the vendor issues during
    the OAuth/file-picker flow. Only these two methods talk to the vendor — swap
    them out and nothing else needs to change.
    """

    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key or os.getenv("UNIFIED_STORAGE_API_KEY")
        self.base_url = base_url or os.getenv("UNIFIED_STORAGE_BASE_URL")

    def list_files(self, token, cursor=None):
        """Yield file metadata dicts: {id, name, mime, modified}.

        `cursor` is a last-sync marker (e.g. a timestamp or page token) so repeat
        syncs pull only new/changed files instead of the whole library.

        TODO(vendor): call the vendor's "list files" endpoint with the connection
        token and map its response to these fields.
        """
        raise NotImplementedError("Implement against the chosen unified-storage vendor")

    def download_file(self, token, file_id):
        """Return the raw bytes of one file.

        TODO(vendor): call the vendor's "download file" endpoint. The bytes are
        held in memory only long enough to extract text — never written to disk.
        """
        raise NotImplementedError("Implement against the chosen unified-storage vendor")


# --------------------------------------------------------------------------- #
# Text extraction — same approach as the manual-upload endpoint.
# --------------------------------------------------------------------------- #
def _extract_text(name, data):
    lower = name.lower()
    if lower.endswith(".pdf"):
        doc = pymupdf.open(stream=data, filetype="pdf")
        text = "\n".join(doc[i].get_text() for i in range(doc.page_count))
        doc.close()
        return text
    # TODO: add .docx / .html extractors as needed. Plain text as a fallback.
    if lower.endswith((".txt", ".md", ".csv")):
        return data.decode("utf-8", errors="ignore")
    return ""   # unknown type -> skip (logged by caller)


# --------------------------------------------------------------------------- #
# Store one document — IDENTICAL scoping to the manual-upload path
# (collection_id + user_id), so connector docs are isolated per company and
# never leak across tenants, exactly like uploads.
# --------------------------------------------------------------------------- #
def _store(conn, collection_id, user_id, title, url, text):
    if not text.strip():
        return 0
    chunks = chunk_text([(1, text)])
    embeddings = embed_chunks(chunks)

    cur = conn.cursor()
    cur.execute("DELETE FROM chunks WHERE url = %s", (url,))   # idempotent re-sync
    for (chunk, _page), embedding in zip(chunks, embeddings):
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
        cur.execute(
            """INSERT INTO chunks
               (content, embedding, source_file, company, subject, issue_date,
                url, source, doc_type, collection_id, user_id)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (chunk, embedding_str, title, title, "", "", url,
             "Connected", "upload", collection_id, user_id),
        )
    conn.commit()
    cur.close()
    return len(chunks)


# --------------------------------------------------------------------------- #
# Sync one connection into its collection.
# --------------------------------------------------------------------------- #
def sync_connection(connection, client=None):
    """Pull new/changed files for one connection and ingest them.

    connection: dict with id, collection_id, user_id, token, cursor.
    Returns (files_ingested, chunks_stored). Incremental via the stored cursor —
    a re-sync only touches files changed since last time.

    Intended to run as a BACKGROUND job (thousands of files), not inside a request.
    """
    client = client or UnifiedStorageClient()
    conn = psycopg2.connect(dbname="regintel")
    files, chunks_total = 0, 0
    latest_cursor = connection.get("cursor")

    for f in client.list_files(connection["token"], cursor=connection.get("cursor")):
        url = f"connector://{connection['id']}/{f['id']}"   # stable id for idempotent re-sync
        try:
            data = client.download_file(connection["token"], f["id"])
            text = _extract_text(f["name"], data)
            n = _store(conn, connection["collection_id"], connection["user_id"],
                       f["name"], url, text)
        except Exception as e:
            print(f"  [connector] skip {f.get('name')}: {e}")
            continue
        if n:
            files += 1
            chunks_total += n
        latest_cursor = f.get("modified") or latest_cursor

    # Advance the cursor so the next sync is incremental.
    cur = conn.cursor()
    cur.execute("UPDATE connections SET cursor = %s, last_synced = now() WHERE id = %s",
                (latest_cursor, connection["id"]))
    conn.commit()
    cur.close()
    conn.close()
    print(f"  [connector] connection {connection['id']}: {files} files, {chunks_total} chunks")
    return files, chunks_total

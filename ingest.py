"""Shared ingestion helpers: chunk text and embed it.

Used by scrape.py's adapters. Document fetching/storage lives in scrape.py.
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def chunk_text(pages, chunk_size=800, overlap=150):
    """Split (page_number, text) pairs into overlapping chunks.

    Returns a list of (chunk_text, page_number) tuples. The overlap keeps a
    sentence that straddles a boundary intact in at least one chunk.
    """
    chunks = []
    for page_number, text in pages:
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append((text[start:end], page_number))
            start = end - overlap
    return chunks


def embed_chunks(chunks):
    """Embed a list of (chunk_text, page_number) pairs in one API call."""
    texts = [chunk for chunk, page in chunks]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]

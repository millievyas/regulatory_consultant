import pymupdf # the PDF library
import os
import glob
import psycopg2

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv() # reads your .env file
client = OpenAI() # automatically uses the API key from your .env file

def extract_text(pdf_path):
    doc = pymupdf.open(pdf_path) # open the PDF file
    pages = [] # collect (page_number, text) pairs

    for page_number in range(doc.page_count):
        page = doc[page_number] # grab one page
        text = page.get_text() # extract text from the page
        pages.append((page_number + 1, text))
    return pages

def chunk_text(pages, chunk_size=800, overlap=150):
    chunks = [] # will hold (chunk_text, page_number) pairs
    for page_number, text in pages:
        start = 0
        while start < len(text):
            end = start + chunk_size # where the chunk ends
            chunk = text[start:end] # slice out the piece of text
            chunks.append((chunk, page_number))
            start = end - overlap
    return chunks

def embed_chunks(chunks):
    texts = [chunk for chunk, page in chunks]
    # api call
    response = client.embeddings.create( 
        model="text-embedding-3-small",
        input=texts
    )

    embeddings = [item.embedding for item in response.data]
    return embeddings

def store_chunks(chunks, embeddings, source_file):
    conn = psycopg2.connect(dbname = "regintel") # connect to the database
    cur = conn.cursor() # a cursor runs SQL commands
    cur.execute("DELETE FROM chunks WHERE source_file = %s", (source_file,))

    for (chunk, page), embedding in zip(chunks, embeddings):
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        cur.execute(
            "INSERT INTO chunks (content, embedding, source_file, page) VALUES (%s, %s, %s, %s)",
            (chunk, embedding_str, source_file, page)
        )

    conn.commit()
    cur.close()
    conn.close()
    print(f"Stored {len(chunks)} chunks in the database")
        
def ingest_file(pdf_path):
    filename = os.path.basename(pdf_path)   # "docs/cgmp_letter.pdf" -> "cgmp_letter.pdf"
    pages = extract_text(pdf_path)
    chunks = chunk_text(pages)
    embeddings = embed_chunks(chunks)
    store_chunks(chunks, embeddings, filename)
    print(f"Ingested {filename}: {len(chunks)} chunks")

def ingest_folder(folder="docs"):
    pdf_paths = glob.glob(os.path.join(folder, "*.pdf"))
    print(f"Found {len(pdf_paths)} PDF(s) in {folder}/")
    for path in pdf_paths:
        ingest_file(path)

if __name__ == "__main__":
    ingest_folder("docs")
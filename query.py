import psycopg2

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# embeds the question into a vector
def embed_query(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",   
        input=[text]
    )

    return response.data[0].embedding

def search(query, top_k=5, source=None, doc_type=None):
    embedding = embed_query(query)
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

    conditions = []
    params = []
    if source:
        conditions.append("source = %s")
        params.append(source)
    if doc_type:
        conditions.append("doc_type = %s")
        params.append(doc_type)
    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    params.append(embedding_str)   # for the <=> similarity operator
    params.append(top_k)           # for LIMIT

    conn = psycopg2.connect(dbname="regintel")
    cur = conn.cursor()
    cur.execute(
        f"""SELECT content,
                   COALESCE(company, source_file) AS company,
                   subject, url, source
            FROM chunks
            {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s""",
        params
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

def list_documents(source=None, doc_type=None, limit=100):
    conditions, params = [], []
    if source:
        conditions.append("source = %s"); params.append(source)
    if doc_type:
        conditions.append("doc_type = %s"); params.append(doc_type)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)

    conn = psycopg2.connect(dbname="regintel")
    cur = conn.cursor()
    cur.execute(f"""
        SELECT DISTINCT COALESCE(company, source_file) AS title, source, doc_type
        FROM chunks {where}
        ORDER BY source, title
        LIMIT %s
    """, params)
    rows = cur.fetchall()
    cur.close(); conn.close()
    return rows

def fetch_document(title, max_chars=12000):
    conn = psycopg2.connect(dbname="regintel")
    cur = conn.cursor()
    cur.execute("""
        SELECT content FROM chunks
        WHERE COALESCE(company, source_file) = %s
        ORDER BY id
    """, (title,))
    rows = cur.fetchall()
    cur.close(); conn.close()
    return "\n".join(r[0] for r in rows)[:max_chars]
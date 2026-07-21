import psycopg2

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


# embeds the question into a vector
def embed_query(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    )
    return response.data[0].embedding


def _scope(collection_ids):
    """Build the SQL scope clause + params for the shared corpus + allowed collections.

    collection_ids=None  -> global reference corpus only (collection_id IS NULL).
    collection_ids=[...]  -> global corpus PLUS those collections (and nothing else).
    """
    if collection_ids:
        return "(collection_id IS NULL OR collection_id = ANY(%s))", [list(collection_ids)]
    return "collection_id IS NULL", []


def search(query, top_k=5, source=None, doc_type=None, collection_ids=None,
           with_scores=False):
    """Semantic search over the corpus.

    with_scores=False -> rows of (content, company, subject, url, source).
    with_scores=True  -> rows of (content, company, subject, url, source, distance),
                         where distance is cosine distance (0 = identical). Rows are
                         ordered nearest-first, so row[0] is the best match — handy
                         for deciding whether the corpus can answer at all.
    """
    embedding = embed_query(query)
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

    scope_sql, where_params = _scope(collection_ids)
    conditions = [scope_sql]
    if source:
        conditions.append("source = %s")
        where_params.append(source)
    if doc_type:
        conditions.append("doc_type = %s")
        where_params.append(doc_type)
    where_clause = "WHERE " + " AND ".join(conditions)

    select_cols = ("content, COALESCE(company, source_file) AS company, "
                   "subject, url, source")
    lead_params = []
    if with_scores:
        select_cols += ", embedding <=> %s::vector AS distance"
        lead_params.append(embedding_str)   # distance column in SELECT

    # Param order must match %s order: SELECT distance, then WHERE, then ORDER BY, LIMIT.
    params = lead_params + where_params + [embedding_str, top_k]

    conn = psycopg2.connect(dbname="regintel")
    cur = conn.cursor()
    cur.execute(
        f"""SELECT {select_cols}
            FROM chunks
            {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s""",
        params,
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results


def list_documents(source=None, doc_type=None, limit=100, collection_ids=None):
    scope_sql, params = _scope(collection_ids)
    conditions = [scope_sql]
    if source:
        conditions.append("source = %s")
        params.append(source)
    if doc_type:
        conditions.append("doc_type = %s")
        params.append(doc_type)
    where = "WHERE " + " AND ".join(conditions)
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
    cur.close()
    conn.close()
    return rows


def fetch_document(title, max_chars=12000, collection_ids=None):
    scope_sql, params = _scope(collection_ids)
    params = [title] + params
    conn = psycopg2.connect(dbname="regintel")
    cur = conn.cursor()
    cur.execute(f"""
        SELECT content FROM chunks
        WHERE COALESCE(company, source_file) = %s AND {scope_sql}
        ORDER BY id
    """, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return "\n".join(r[0] for r in rows)[:max_chars]

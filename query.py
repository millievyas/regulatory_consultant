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

def search(query, top_k=5):
    embedding = embed_query(query)
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

    conn = psycopg2.connect(dbname="regintel")
    cur = conn.cursor()
    cur.execute(
        """SELECT content,
                  COALESCE(company, source_file) AS company,
                  subject,
                  url
           FROM chunks
           ORDER BY embedding <=> %s::vector
           LIMIT %s""",
        (embedding_str, top_k)
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

def answer_question(query, top_k=5):
    results = search(query, top_k)

    context = "\n\n".join(
        f"[Company: {company} | Subject: {subject}]\n{content}"
        for content, company, subject, url in results
    )

    system_prompt = (
        "You are a regulatory analyst specializing in FDA compliance. "
        "Answer the user's question using ONLY the provided context. "
        "If the answer is not in the context, say you cannot find it. "
        "Cite the company name for each claim you make."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print("Ask questions about your documents. Type 'quit' to exit.")
    while True:
        question = input("\nQuestion: ")

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not question.strip():
            continue

        answer = answer_question(question)
        print("\n" + answer)
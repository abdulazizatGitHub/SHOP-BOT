import pandas as pd
from sentence_transformers import SentenceTransformer
import psycopg2
import os

# ENV or configure here
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB = os.getenv("PG_DB", "shopbot")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASS", "")

CSV_PATH = "dataset/Chatbot_Dataset.csv"
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

print("Loading embedding model:", EMBED_MODEL)
model = SentenceTransformer(EMBED_MODEL)

def connect_db():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS
    )

def embed(texts):
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

def ingest():
    # Try utf-8 first, fallback to latin-1
    try:
        df = pd.read_csv(CSV_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        print("⚠️ UTF-8 failed, retrying with latin-1 encoding...")
        df = pd.read_csv(CSV_PATH, encoding="latin-1")

    df = df.dropna(subset=["Question/Trigger", "Answer/Response"])
    df = df.drop_duplicates(subset=["Question/Trigger", "Answer/Response"])

    texts = (
        df["Question/Trigger"].astype(str) + " ||| " + df["Answer/Response"].astype(str)
    ).tolist()
    print(f"Embedding {len(texts)} docs...")
    embeddings = embed(texts)  # numpy array (n, 384)

    conn = connect_db()
    cur = conn.cursor()

    for i, row in df.reset_index().iterrows():
        faq_id = f"faq-{i}"
        q = str(row["Question/Trigger"]).strip()
        a = str(row["Answer/Response"]).strip()
        typ = str(row.get("Type", "")).strip()
        emb = embeddings[i].tolist()
        cur.execute(
            """
            INSERT INTO faqs (faq_id, question, answer, type, embedding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (faq_id) DO UPDATE
            SET question = EXCLUDED.question,
                answer = EXCLUDED.answer,
                type = EXCLUDED.type,
                embedding = EXCLUDED.embedding;
            """,
            (faq_id, q, a, typ, emb),
        )
        if i % 100 == 0:
            conn.commit()
            print(f"Inserted {i} rows...")

    conn.commit()
    cur.close()
    conn.close()
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest()

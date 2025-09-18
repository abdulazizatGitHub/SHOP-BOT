# chat.py
import psycopg2
import requests
import os
import json

from dotenv import load_dotenv
load_dotenv()


PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB   = os.getenv("PG_DB", "shopbot")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASS", "")
MODEL_SERVER = os.getenv("MODEL_SERVER_URL", "http://127.0.0.1:8001")

def connect_db():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS
    )

def embed(text):
    resp = requests.post(f"{MODEL_SERVER}/embed", json={"text": text})
    return resp.json()["embedding"]

def generate(prompt):
    resp = requests.post(f"{MODEL_SERVER}/generate", json={"text": prompt})
    return resp.json().get("text", "")

def query_faqs(query, k=3):
    vec = embed(query)
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT question, answer FROM faqs ORDER BY embedding <-> %s::vector LIMIT %s;",
        (vec, k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def chat():
    print("SHOP-BOT CLI — type 'quit' to exit")
    while True:
        user_in = input("You: ")
        if user_in.lower() in ["quit", "exit"]:
            break
        docs = query_faqs(user_in, k=3)
        context = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in docs])
        prompt = f"""
You are a helpful shop assistant. Use the FAQ context below to answer the user.
If not enough info, politely say you don't know.

FAQ context:
{context}

User: {user_in}
Bot:"""
        bot_out = generate(prompt)
        print("Bot:", bot_out.strip())

if __name__ == "__main__":
    chat()

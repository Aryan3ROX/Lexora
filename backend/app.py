from typing import Union
from fastapi import FastAPI, Request, Body
from ai.utils import cosine_similarity, lex_response
from sentence_transformers import SentenceTransformer
import os
import psycopg2
import numpy as np
from dotenv import load_dotenv
import ast
from transformers import pipeline

load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')

# Load models
embedder = SentenceTransformer('./ai/embedding/artifacts/fine_tuned_book_embedder')
lex = pipeline("text2text-generation", model="./ai/lex_model", device=-1)

# Retrieval of all book data
conn = psycopg2.connect(
    dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
)
with conn.cursor() as cur:
    cur.execute("SELECT id, title, author, embedding, summary FROM books;")
    rows = cur.fetchall()

titles = []
authors = []
summaries = []
embeddings = []
for row in rows:
    titles.append(row[1])
    authors.append(row[2])
    emb = row[3]
    if isinstance(emb, str):
        emb = ast.literal_eval(emb)
    emb = np.array(emb, dtype=np.float32)
    embeddings.append(emb)
    summaries.append(row[4] if len(row) > 4 else "")
embeddings = np.stack(embeddings)

print("Starting app!")
app = FastAPI()

@app.get('/')
def read_root():
    return {'Hello':'World'}

@app.post('/search')
def search_query(data: dict = Body(...)):
    user_query = data["user_query"]
    use_lex = data.get("lex", False)
    query_embedding = embedder.encode(user_query, convert_to_tensor=True).cpu().numpy()
    scores = cosine_similarity(query_embedding, embeddings)
    top_idx = np.argsort(scores)[::-1][:50]
    results = []
    for idx in top_idx:
        results.append({
            "title": titles[idx],
            "author": authors[idx],
            "summary": summaries[idx],
        })
    explanations = []
    if use_lex:
        indices = top_idx[:5]
        explanations = lex_response(lex, user_query, titles, authors, summaries, indices)
        for i, idx in enumerate(indices):
            pos = list(top_idx).index(idx)
            results[pos]["explanation"] = explanations[i]
    return {"results": results, "top_idx": top_idx.tolist()}

@app.post('/explanations')
def get_explanations(data: dict = Body(...)):
    user_query = data["user_query"]
    top_idx = data["top_idx"]
    start = data["start"]
    end = data["end"]
    indices = top_idx[start:end]
    explanations = lex_response(lex, user_query, titles, authors, summaries, indices)
    results = []
    for i, idx in enumerate(indices):
        results.append({
            "title": titles[idx],
            "author": authors[idx],
            "summary": summaries[idx],
            "explanation": explanations[i]
        })
    return {"results": results}
    
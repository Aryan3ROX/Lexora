from typing import Union
from fastapi import FastAPI, Request, Body
from ai.utils import cosine_similarity, lex_response, syno_response, get_search_vectors, rexy_response
from sentence_transformers import SentenceTransformer
import os
import psycopg2
import numpy as np
from dotenv import load_dotenv
import ast

load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')

# Load models
embedder = SentenceTransformer('./ai/embedding/artifacts/fine_tuned_book_embedder')
lex_path = "./ai/lex_model"

# Retrieval of all book data
conn = psycopg2.connect(
    dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
)
with conn.cursor() as cur:
    cur.execute("SELECT id, title, author, embedding, summary FROM books;")
    rows = cur.fetchall()

book_ids = []
titles = []
authors = []
summaries = []
embeddings = []
for row in rows:
    book_ids.append(row[0])
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
            "id": book_ids[idx],
            "title": titles[idx],
            "author": authors[idx],
            "summary": summaries[idx],
            "scores": float(scores[idx])
        })
    explanations = []
    if use_lex:
        indices = top_idx[:5]
        explanations = lex_response(lex_path, user_query, titles, authors, summaries, indices)
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
    explanations = lex_response(lex_path, user_query, titles, authors, summaries, indices)
    results = []
    for i, idx in enumerate(indices):
        results.append({
            "title": titles[idx],
            "author": authors[idx],
            "summary": summaries[idx],
            "explanation": explanations[i]
        })
    return {"results": results}

@app.post('/synopsis')
def get_synopsis(data: dict = Body(...)):
    book_id = data["book_id"]
    with conn.cursor() as cur:
        cur.execute("SELECT title, author, summary FROM books WHERE id = %s", (book_id,))
        row = cur.fetchone()
        title, author, summary = row
    synopsis = syno_response(lex_path, title, author, summary)
    return {"synopsis": synopsis}

@app.post('/recommendations')
def get_recommendations(data: dict = Body(...)):
    user_id = data["user_id"]
    with conn.cursor() as cur:
        cur.execute("""
            SELECT b.embedding, 
            CASE 
                WHEN ub.status = 'plan_to_read' THEN 5 
                ELSE ub.rating 
            END as effective_rating
            FROM user_books ub
            JOIN books b ON ub.book_id = b.id
            WHERE ub.user_id = %s 
            AND (
              (ub.status = 'read' AND ub.rating > 0)
              OR 
              (ub.status = 'plan_to_read')
            )
        """, (user_id,))
        book_data = cur.fetchall()
        cur.execute("SELECT book_id FROM user_books WHERE user_id = %s", (user_id,))
        excluded_ids = {r[0] for r in cur.fetchall()}
    if not book_data:
        return {"results": [], "message": "Start reading books to get your customized recommendations!"}
    
    book_data = [
        (np.array(ast.literal_eval(row[0]), dtype=np.float32), row[1])
        if isinstance(row[0], str) else (row[0], row[1])
        for row in book_data
    ]
    search_vectors = get_search_vectors(book_data)
    results = rexy_response(search_vectors,book_ids,excluded_ids,embeddings)
    return {"results": results}        
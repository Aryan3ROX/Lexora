import pandas as pd
import torch
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2.extras import execute_values
import numpy as np
import ast
import json
from tqdm import tqdm
import os
from dotenv import load_dotenv
load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')

CSV_PATH = 'ai/booksummaries/book_data_clean.csv'
EMBEDDINGS_PATH = 'ai/embedding/artifacts/embedder/book_embeddings.pt'

print("Loading and aligning data...")
df_original = pd.read_csv(CSV_PATH)
loaded = torch.load(EMBEDDINGS_PATH, map_location='cpu')
embeddings_tensor = loaded.get("embeddings")
meta = loaded.get("meta")

if embeddings_tensor is None:
    raise ValueError("Embeddings not found in the loaded file (expected key 'embeddings').")

if not isinstance(embeddings_tensor, torch.Tensor):
    embeddings_tensor = torch.as_tensor(embeddings_tensor)

embeddings_np = embeddings_tensor.cpu().numpy()
if meta and "titles" in meta and len(meta["titles"]) != len(embeddings_np):
    print("Warning: meta titles count differs from embeddings count.")

df_clean = df_original.dropna(subset=['title', 'author', 'genres', 'summary']).reset_index(drop=True)
if 'pub_date' in df_clean.columns:
    df_clean['pub_date'] = pd.to_datetime(df_clean['pub_date'], errors='coerce').dt.date
    df_clean['pub_date'] = df_clean['pub_date'].where(pd.notna(df_clean['pub_date']), None)
assert len(df_clean) == len(embeddings_np), "Data mismatch after cleaning!"
print(f"Data aligned. Processing {len(df_clean)} books.")

print("\n--- Setting up database ---")
conn = None
try:
    conn = psycopg2.connect(dbname='postgres', user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    
    with conn.cursor() as cur:
        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
        exists = cur.fetchone()
        if not exists:
            print(f"Database '{DB_NAME}' does not exist. Creating it...")
            cur.execute(f"CREATE DATABASE {DB_NAME}")
            print("Database created successfully.")
        else:
            print(f"Database '{DB_NAME}' already exists.")

finally:
    if conn:
        conn.close()

print(f"\n--- Populating '{DB_NAME}' database ---")
try:
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    with conn.cursor() as cur:
        print("Creating pgvector extension and 'books' table...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY,
                title TEXT,
                author TEXT,
                pub_date DATE,
                genres JSONB,
                summary TEXT,
                cover_image_url TEXT,
                embedding vector(768)
            );
        """)
        
        print("Preparing data for insertion...")
        data_to_insert = []
        for i, row in tqdm(df_clean.iterrows(), total=df_clean.shape[0]):
            embedding = embeddings_np[i]
            pub_date = row.get('pub_date')
            genres_str = row.get('genres')
            genres_json = None
            if pd.notna(genres_str):
                try:
                    genres_list = ast.literal_eval(str(genres_str))
                    genres_json = json.dumps(genres_list)
                except (ValueError, SyntaxError):
                    genres_json = None
            
            data_to_insert.append((
                int(row.get('id')), row.get('title'), row.get('author'), pub_date, genres_json,
                row.get('summary'), row.get('cover_image_url'), embedding.tolist()
            ))

        print(f"Inserting {len(data_to_insert)} records into the database...")
        execute_values(
            cur,
            """
            INSERT INTO books (id, title, author, pub_date, genres, summary, cover_image_url, embedding)
            VALUES %s
            """,
            data_to_insert
        )
        conn.commit()
    
    print("Data insertion complete.")
    
    with conn.cursor() as cur:
        print('Creating users table.')
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                fullname TEXT,   
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP        
            )
        """)
        
        print('Creating user_books table.')
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_books (
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                book_id INTEGER REFERENCES books(id) ON DELETE CASCADE,
                status TEXT CHECK (status IN ('read', 'reading', 'plan_to_read', 'no_plan_to_read')),
                rating INTEGER CHECK (rating >= 0 AND rating <= 10),
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, book_id)
            )
        """)
        conn.commit()
        print("Tables Created.")

except psycopg2.Error as e:
    print(f"Database error: {e}")

finally:
    if conn:
        conn.close()
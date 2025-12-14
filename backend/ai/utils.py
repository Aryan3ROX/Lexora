import numpy as np
from transformers import pipeline

def cosine_similarity(query, matrix):
    query_norm = query / np.linalg.norm(query)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.dot(matrix_norm,query_norm)

def lex_response(generator_path, query, titles, authors, summaries, indices):
    explanations = []
    for idx in indices:
        title = titles[idx]
        author = authors[idx]
        summary = summaries[idx]
        prompt = (
            "Explain in less than 50 words how this book is relevant to the query."
            f'User query: "{query}"\n'
            f'Book: {title} by {author}\n'
            f'Summary: {summary}\n'            
        )
        generator = pipeline("text2text-generation", model=generator_path, device=-1)
        result = generator(prompt, max_new_tokens=512)
        explanation = result[0]['generated_text'].strip()
        explanations.append(explanation)
    return explanations

def syno_response(generator_path, title, author, summary):
    synopsis = None
    prompt = (
        f'Given the summary of the book {title} by {author}, generate a non-spoiler synopsis of the book in less than 60 words.\nSummary: {summary}'
    )
    generator = pipeline("text2text-generation", model=generator_path, device=-1)
    result = generator(prompt, max_new_tokens=512)
    synopsis = result[0]['generated_text'].strip()
    return synopsis
    
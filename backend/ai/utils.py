import numpy as np

def cosine_similarity(query, matrix):
    query_norm = query / np.linalg.norm(query)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.dot(matrix_norm,query_norm)

def lex_response(generator, query, titles, authors, summaries, indices):
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
        result = generator(prompt, max_new_tokens=256)
        explanation = result[0]['generated_text'].strip()
        explanations.append(explanation)
    return explanations
import numpy as np
from transformers import pipeline

def cosine_similarity(query, matrix):
    query_norm = query / np.linalg.norm(query)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.dot(matrix_norm,query_norm)

def lex_response(generator_path, query, title, author, summary):
    prompt = (
        "Explain in less than 50 words how this book is relevant to the query."
        f'User query: "{query}"\n'
        f'Book: {title} by {author}\n'
        f'Summary: {summary}\n'            
    )
    generator = pipeline("text2text-generation", model=generator_path, device=-1)
    result = generator(prompt, max_new_tokens=512)
    explanation = result[0]['generated_text'].strip()
    return explanation

def syno_response(generator_path, title, author, summary):
    synopsis = None
    prompt = (
        f'Based on the summary, give a short, spoiler-free overview (max 60 words) of what a reader will experience when reading "{title}" by {author}. Highlight the tone, themes, or style, but avoid revealing specific plot details.\nSummary: {summary}'
    )
    generator = pipeline("text2text-generation", model=generator_path, device='cpu')
    result = generator(prompt, max_new_tokens=512)
    synopsis = result[0]['generated_text'].strip()
    return synopsis
    
def get_search_vectors(book_data):
    user_history = [(np.array(row[0]), row[1]) for row in book_data]
    clusters = []
    for item in user_history:
        emb, rating = item
        best_cluster_index = -1
        best_similarity = -1.0
        for i, cluster in enumerate(clusters):
            cluster_embeddings = [c[0] for c in cluster]
            cluster_centroid = np.mean(cluster_embeddings, axis=0)
            
            similarity = np.dot(emb, cluster_centroid) / (np.linalg.norm(emb) * np.linalg.norm(cluster_centroid))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster_index = i
                
        if best_similarity > 0.6:
            clusters[best_cluster_index].append(item)
        else:
            clusters.append([item])
            
    search_vectors = []
    for cluster in clusters:
        weighted_sum = np.zeros_like(cluster[0][0])
        total_weight = 0
        
        for emb, rating in cluster:
            weight = rating
            total_weight += weight
            weighted_sum += emb * weight
        
        if total_weight > 0:
            search_vectors.append(weighted_sum/total_weight)
            
    return search_vectors

def rexy_response(search_vectors, book_ids, excluded_ids, embeddings):
    candidate_pool = {}
    
    for sv in search_vectors:
        scores = cosine_similarity(sv, embeddings)
        top_idx = np.argsort(scores)[::-1][:100]
        
        for idx in top_idx:
            real_book_id = book_ids[idx]
            if real_book_id in excluded_ids:
                continue
            score = float(scores[idx])
            
            if idx not in candidate_pool or score > candidate_pool[idx]["score"]:
                candidate_pool[idx] = {
                    "id": real_book_id,
                    "score": score,
                }
    
    all_candidates = list(candidate_pool.values())
    all_candidates.sort(key=lambda x: x["score"], reverse=True)
    
    results = all_candidates[:100]
    return results
import numpy as np


def normalize_scores(scores):
    scores = np.array(scores, dtype=float)

    if len(scores) == 0:
        return scores

    min_score = scores.min()
    max_score = scores.max()

    if max_score == min_score:
        if max_score == 0:
            return np.zeros_like(scores)

        return np.ones_like(scores)

    return (scores - min_score) / (max_score - min_score)


def hybrid_search(
    query_vector,
    query,
    vector_store,
    keyword_retriever,
    top_k=3,
    vector_weight=0.5,
):
    keyword_weight = 1 - vector_weight

    vector_results = vector_store.search(
        query_vector,
        top_k=100,
    )

    keyword_results = keyword_retriever.search(
        query,
        top_k=100,
    )

    all_results = {}

    for result in vector_results:
        key = (
            str(result["document_id"]),
            result["chunk_id"],
        )

        all_results[key] = {
            "document_id": result["document_id"],
            "chunk_id": result["chunk_id"],
            "chunk": result["chunk"],
            "vector_score": result["score"],
            "keyword_score": 0.0,
        }

    for result in keyword_results:
        key = (
            str(result["document_id"]),
            result["chunk_id"],
        )

        if key not in all_results:
            all_results[key] = {
                "document_id": result["document_id"],
                "chunk_id": result["chunk_id"],
                "chunk": result["chunk"],
                "vector_score": 0.0,
                "keyword_score": result["score"],
            }
        else:
            all_results[key]["keyword_score"] = result["score"]

    if not all_results:
        return []

    vector_scores = [
        item["vector_score"]
        for item in all_results.values()
    ]

    keyword_scores = [
        item["keyword_score"]
        for item in all_results.values()
    ]

    normalized_vector = normalize_scores(vector_scores)
    normalized_keyword = normalize_scores(keyword_scores)

    results = []

    for i, item in enumerate(all_results.values()):
        score = (
            vector_weight * normalized_vector[i]
            + keyword_weight * normalized_keyword[i]
        )

        results.append({
            "document_id": item["document_id"],
            "chunk_id": item["chunk_id"],
            "chunk": item["chunk"],
            "vector_score": item["vector_score"],
            "keyword_score": item["keyword_score"],
            "score": float(score),
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return results[:top_k]
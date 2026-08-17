import os

from dotenv import load_dotenv
from openai import OpenAI

from rag.embeddings import get_embedding
from rag.supabase_store import SupabaseVectorStore
from rag.keyword_search import KeywordRetriever
from rag.hybrid_search import hybrid_search
from rag.evaluation.retrieval_dataset import TEST_DATA


load_dotenv()


def recall_at_k(retrieved_ids, relevant_ids):

    retrieved_ids = set(retrieved_ids)
    relevant_ids = set(relevant_ids)

    if not relevant_ids:
        return 0

    return int(bool(retrieved_ids & relevant_ids))


def evaluate_vector_search(client, store):

    scores = []

    for test in TEST_DATA:

        question = test["question"]
        relevant_ids = test["relevant_chunk_ids"]

        query_embedding = get_embedding(
            client,
            question
        )

        results = store.search(
            query_embedding,
            top_k=3
        )

        retrieved_ids = [
            result["chunk_id"]
            for result in results
        ]

        scores.append(
            recall_at_k(
                retrieved_ids,
                relevant_ids
            )
        )

    return scores


def evaluate_keyword_search(retriever):

    scores = []

    for test in TEST_DATA:

        question = test["question"]
        relevant_ids = test["relevant_chunk_ids"]

        results = retriever.search(
            question,
            top_k=3
        )

        retrieved_ids = [
            result["chunk_id"]
            for result in results
        ]

        scores.append(
            recall_at_k(
                retrieved_ids,
                relevant_ids
            )
        )

    return scores


def evaluate_hybrid_search(
    client,
    store,
    keyword_retriever
):

    scores = []

    for test in TEST_DATA:

        question = test["question"]
        relevant_ids = test["relevant_chunk_ids"]

        query_embedding = get_embedding(
            client,
            question
        )

        results = hybrid_search(
            query_vector=query_embedding,
            query=question,
            vector_store=store,
            keyword_retriever=keyword_retriever,
            top_k=3
        )

        retrieved_ids = [
            result["chunk_id"]
            for result in results
        ]

        scores.append(
            recall_at_k(
                retrieved_ids,
                relevant_ids
            )
        )

    return scores


def calculate_recall(scores):

    if not scores:
        return 0

    return sum(scores) / len(scores)


def main():

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    store = SupabaseVectorStore()

    keyword_retriever = KeywordRetriever(
        store.get_chunks_for_keyword_search()
    )

    vector_scores = evaluate_vector_search(
        client,
        store
    )

    keyword_scores = evaluate_keyword_search(
        keyword_retriever
    )

    hybrid_scores = evaluate_hybrid_search(
        client,
        store,
        keyword_retriever
    )

    vector_recall = calculate_recall(
        vector_scores
    )

    keyword_recall = calculate_recall(
        keyword_scores
    )

    hybrid_recall = calculate_recall(
        hybrid_scores
    )

    print(
        f"Vector Search Hit@3: "
        f"{vector_recall:.2f}"
    )

    print(
        f"Keyword Search Hit@3: "
        f"{keyword_recall:.2f}"
    )

    print(
        f"Hybrid Search Hit@3: "
        f"{hybrid_recall:.2f}"
    )


if __name__ == "__main__":
    main()
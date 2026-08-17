import os

from dotenv import load_dotenv
from openai import OpenAI

from rag.embeddings import get_embedding
from rag.vector_store import VectorStore
from rag.keyword_search import KeywordRetriever
from rag.hybrid_search import hybrid_search 

from rag.evaluation.retrieval_dataset import TEST_DATA


load_dotenv()

INDEX_PATH = "storage/index.faiss"
CHUNKS_PATH = "storage/chunks.pkl"


def recall_at_k(retrieved_ids, relevant_ids):

    retrieved_ids = set(retrieved_ids)
    relevant_ids = set(relevant_ids)

    if not relevant_ids:
        return 0

    return int(
        bool(retrieved_ids & relevant_ids)
    )


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

        score = recall_at_k(
            retrieved_ids,
            relevant_ids
        )

        scores.append(score)

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

        score = recall_at_k(
            retrieved_ids,
            relevant_ids
        )

        scores.append(score)

        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print(f"Relevant: {relevant_ids}")
        print(f"Keyword retrieved: {retrieved_ids}")

        for result in results:

            print(
                f"\nChunk {result['chunk_id']} "
                f"Score: {result['score']:.4f}"
            )

            print(result["chunk"])

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

        # Create query embedding
        query_embedding = get_embedding(
            client,
            question
        )

        # Hybrid retrieval
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

        score = recall_at_k(
            retrieved_ids,
            relevant_ids
        )

        scores.append(score)

        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print(f"Relevant: {relevant_ids}")
        print(f"Hybrid retrieved: {retrieved_ids}")

        for result in results:

            print(
                f"\nChunk {result['chunk_id']} "
                f"Hybrid: {result['score']:.4f} "
                f"Vector: {result['vector_score']:.4f} "
                f"Keyword: {result['keyword_score']:.4f}"
            )

            print(result["chunk"])

    return scores

def main():

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    store = VectorStore.load(
        INDEX_PATH,
        CHUNKS_PATH
    )

    # Use the chunks already stored in FAISS
    keyword_retriever = KeywordRetriever(
        store.chunks
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

    vector_recall = (
        sum(vector_scores)
        / len(vector_scores)
    )

    keyword_recall = (
        sum(keyword_scores)
        / len(keyword_scores)
    )

    hybrid_recall = (
    sum(hybrid_scores)
    / len(hybrid_scores)
    )

    print("\n" + "=" * 60)

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

    print("=" * 60)


if __name__ == "__main__":
    main()
import os

from dotenv import load_dotenv
from openai import OpenAI

from rag.vector_store import VectorStore
from rag.keyword_search import KeywordRetriever
from rag.rag_pipeline import generate_rag_answer


load_dotenv()


INDEX_PATH = "storage/index.faiss"
CHUNKS_PATH = "storage/chunks.pkl"

TEST_QUESTIONS = [
    {
        "question": "What operating system does the device use?",
        "expected": "Linux based OpenWrt OS"
    },
    {
        "question": "What is the operating voltage of the device?",
        "expected": "8 - 32V DC"
    },
    {
        "question": "How many SIM slots does the device have?",
        "expected": "2"
    },
    {
        "question": "What is the weight of the device?",
        "expected": "200g"
    },
    {
        "question": "What type of cellular connectivity does the device support?",
        "expected": "4G LTE Cat 4, 3G, 2G"
    },
    {
        "question": "What WiFi standard does the device support?",
        "expected": "IEEE 802.11"
    },
    {
        "question": "How many Ethernet ports does the device have?",
        "expected": "2"
    },
    {
        "question": "What is the operating temperature range?",
        "expected": "-30°C to +70°C"
    },
    {
        "question": "What remote management methods are supported?",
        "expected": "WEB UI, SSH, SMS, FOTA"
    },
    {
        "question": "What is the price of the device?",
        "expected": "I don't know"
    },
    {
        "question": "Does the document mention GPS functionality?",
        "expected": "I don't know"
    },
    {
        "question": "What is the device's maximum LTE speed?",
        "expected": "150 Mbps"
    }
]

PROMPTS = {

    "basic": """
Answer the question using the provided context.
""",

    "grounded": """
Answer the question using only the provided context.

If the answer is not in the context,
say "I don't know."

Do not use outside knowledge.
""",

    "technical": """
You are a technical documentation assistant.

Answer using only the provided context.

Rules:
- Do not use outside knowledge.
- Do not invent information.
- Prefer exact values and specifications from the context.
- If the answer cannot be found in the context,
  say "I don't know."
- Keep the answer concise and directly answer
  the question.
"""
}


def generate_answer(
    client,
    question,
    context,
    prompt
):

    full_prompt = f"""
{prompt}

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": full_prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content

def judge_answer(
    client,
    question,
    expected,
    answer
):

    judge_prompt = f"""
You are evaluating an answer produced by a RAG chatbot.

Determine whether the generated answer is factually
correct compared with the expected answer.

The wording does NOT need to be identical.
Paraphrases are acceptable.

A response is CORRECT if:
- it contains the correct information
- it does not contradict the expected answer
- it answers the question directly

A response is INCORRECT if:
- the information is wrong
- important information is missing
- it contradicts the expected answer
- it invents information when the expected answer is "I don't know"

Question:
{question}

Expected answer:
{expected}

Generated answer:
{answer}

Return ONLY:
1

if the answer is correct.

Return ONLY:
0

if the answer is incorrect.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": judge_prompt
            }
        ],
        temperature=0
    )

    result = response.choices[0].message.content.strip()

    if result == "1":
        return 1

    return 0

def main():

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    store = VectorStore.load(
        INDEX_PATH,
        CHUNKS_PATH
    )

    keyword_retriever = KeywordRetriever(
        store.chunks
    )

    from rag.embeddings import get_embedding
    from rag.hybrid_search import hybrid_search

    for prompt_name, prompt in PROMPTS.items():

        print("\n" + "=" * 70)

        print(
            f"PROMPT: {prompt_name.upper()}"
        )

        print("=" * 70)

        scores = []

        for test in TEST_QUESTIONS:

            question = test["question"]

            # Create query embedding
            query_vector = get_embedding(
                client,
                question
            )

            # Hybrid retrieval
            results = hybrid_search(
                query_vector=query_vector,
                query=question,
                vector_store=store,
                keyword_retriever=keyword_retriever,
                top_k=3
            )

            # Build context
            context = "\n\n".join(
                result["chunk"]
                for result in results
            )

            # Generate answer
            answer = generate_answer(
                client,
                question,
                context,
                prompt
            )

            print("\nQuestion:")
            print(question)

            print("\nExpected:")
            print(test["expected"])

            print("\nAnswer:")
            print(answer)

            score = judge_answer(
                client,
                question,
                test["expected"],
                answer
            )

            scores.append(score)

            print(
                f"Judge score: {score}"
            )
            total_score = sum(scores)

            average_score = (
            total_score / len(scores)
            )

            print("\n" + "-" * 70)

            print(
            f"{prompt_name.upper()} SCORE: "
            f"{total_score}/{len(scores)} "
            f"({average_score:.2f})"
            )
            print(
                f"Score: {score}"
            )

            print("-" * 70)



if __name__ == "__main__":
    main()
import os

from dotenv import load_dotenv
from openai import OpenAI

from rag.embeddings import get_embedding
from rag.supabase_store import SupabaseVectorStore
from rag.keyword_search import KeywordRetriever
from rag.hybrid_search import hybrid_search


load_dotenv()


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

If the answer is not in the context, say "I don't know."

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
- Keep the answer concise and directly answer the question.
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

Paraphrases are acceptable.

A response is CORRECT if:
- it contains the correct information
- it does not contradict the expected answer
- it answers the question directly

A response is INCORRECT if:
- the information is wrong
- important information is missing
- it contradicts the expected answer
- it invents information when the expected answer is
  "I don't know"

Question:
{question}

Expected answer:
{expected}

Generated answer:
{answer}

Return only 1 if correct.
Return only 0 if incorrect.
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

    return int(
        response.choices[0].message.content.strip() == "1"
    )


def evaluate_prompt(
    client,
    store,
    keyword_retriever,
    prompt
):

    scores = []

    for test in TEST_QUESTIONS:

        question = test["question"]

        query_vector = get_embedding(
            client,
            question
        )

        results = hybrid_search(
            query_vector=query_vector,
            query=question,
            vector_store=store,
            keyword_retriever=keyword_retriever,
            top_k=3
        )

        context = "\n\n".join(
            result["chunk"]
            for result in results
        )

        answer = generate_answer(
            client,
            question,
            context,
            prompt
        )

        score = judge_answer(
            client,
            question,
            test["expected"],
            answer
        )

        scores.append(score)

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

    for prompt_name, prompt in PROMPTS.items():

        score = evaluate_prompt(
            client,
            store,
            keyword_retriever,
            prompt
        )

        print(
            f"{prompt_name}: "
            f"{score:.2f}"
        )


if __name__ == "__main__":
    main()
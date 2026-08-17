import os
import tiktoken

from openai import OpenAI
from dotenv import load_dotenv

from rag.supabase_store import SupabaseVectorStore
from rag.keyword_search import KeywordRetriever
from rag.pipeline import (
    generate_rag_answer,
    should_summarize,
    summarize_conversation,
    update_memory,
    count_message_tokens
)

from config import (
    MODEL
)


load_dotenv()




def main():

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Load vector store
    store = SupabaseVectorStore()

    # Create keyword retriever
    keyword_retriever = KeywordRetriever(
        store.chunks
    )

    # Conversation state
    messages = []
    conversation_summary = ""
    conversation_memory = ""
    message_count = 0

    encoding = tiktoken.encoding_for_model(
        MODEL
    )

    print("=" * 60)
    print("RAG Technical Documentation Assistant")
    print("=" * 60)
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            break

        # Add user message
        messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        # Summarize old conversation if necessary
        if should_summarize(
            messages,
            encoding
        ):

            old_messages = messages[:-4]

            if old_messages:

                print(
                    "\n[Summarizing conversation...]\n"
                )

                conversation_summary = (
                    summarize_conversation(
                        client,
                        old_messages,
                        conversation_summary
                    )
                )

                messages = messages[-4:]

        # Generate answer
        print("\nAI: ", end="")

        answer, results = generate_rag_answer(
            client,
            question,
            store,
            keyword_retriever,
            messages=messages,
            conversation_summary=conversation_summary,
            conversation_memory=conversation_memory,
            top_k=3
        )

        # Save assistant response
        if answer:

            messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        # Update memory every 3 exchanges
        message_count += 1

        if message_count % 3 == 0:

            conversation_memory = update_memory(
                client,
                messages,
                conversation_memory
            )

        # -----------------------------------------
        # SOURCES
        # -----------------------------------------

        print("\n" + "-" * 60)
        print("Sources")
        print("-" * 60)

        for i, result in enumerate(
            results,
            1
        ):

            chunk_id = result.get(
                "chunk_id",
                "unknown"
            )

            score = result.get(
                "score",
                0
            )

            print(
                f"\nSource {i} "
                f"(Chunk {chunk_id}, "
                f"score={score:.4f})"
            )

            # Show shortened source
            chunk_text = result["chunk"]

            if len(chunk_text) > 500:
                chunk_text = (
                    chunk_text[:500]
                    + "..."
                )

            print(chunk_text)

        # -----------------------------------------
        # TOKEN INFORMATION
        # -----------------------------------------

        current_tokens = count_message_tokens(
            messages,
            encoding
        )

        print(
            "\nConversation tokens: "
            f"{current_tokens}"
        )

        print("-" * 60)


if __name__ == "__main__":
    main()
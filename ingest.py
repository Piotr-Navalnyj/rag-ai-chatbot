import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from rag.chunking import chunk_text
from rag.embeddings import get_embedding
from rag.supabase_store import SupabaseVectorStore


load_dotenv()


def main():
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = os.getenv("PDF_PATH")

        if not pdf_path:
            raise ValueError(
                "No PDF specified.\n"
                "Use:\n"
                'python ingest.py "data/your-file.pdf"\n'
                "or set PDF_PATH in .env"
            )

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print("=" * 60)
    print("DOCUMENT INGESTION")
    print("=" * 60)
    print(f"\nPDF: {pdf_path}")

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    print("\n[1/4] Extracting text...")

    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        raise ValueError("No text could be extracted from PDF.")

    print(f"Extracted {len(text)} characters.")

    print("\n[2/4] Creating chunks...")

    chunks = chunk_text(text)

    print(f"Created {len(chunks)} chunks.")

    print("\n[3/4] Creating embeddings...")

    embeddings = []

    for i, chunk in enumerate(chunks):
        print(
            f"Embedding {i + 1}/{len(chunks)}",
            end="\r"
        )

        embedding = get_embedding(client, chunk)
        embeddings.append(embedding)

    print(f"\nCreated {len(embeddings)} embeddings.")

    print("\n[4/4] Uploading to Supabase...")

    store = SupabaseVectorStore()
    store.clear()
    store.add(chunks, embeddings)

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Document: {pdf_path}")
    print(f"Chunks stored: {len(chunks)}")
    print(f"Embeddings stored: {len(embeddings)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
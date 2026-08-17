from rag.chunking import chunk_text


text = """
Python is a programming language.

It is widely used for data science and machine learning.

RAG systems can use Python to process documents.

Documents can be divided into smaller chunks.

Chunks can then be converted into embeddings.
"""


chunks = chunk_text(
    text,
    chunk_size=100
)


for i, chunk in enumerate(chunks):

    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)
import fitz

from rag.chunking import chunk_text
from rag.embeddings import get_embedding


class DocumentService:

    def __init__(self, client, store):
        self.client = client
        self.store = store

    def process_pdf(self, uploaded_file):

        filename = uploaded_file.name

        pdf_bytes = uploaded_file.getvalue()

        if not pdf_bytes:
            raise ValueError("Uploaded PDF is empty.")

        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        text = ""

        for page in document:
            text += page.get_text()
            text += "\n"

        document.close()

        if not text.strip():
            raise ValueError(
                "Could not extract text from PDF."
            )

        chunks = chunk_text(
            text,
            chunk_size=200,
            overlap=50
        )

        if not chunks:
            raise ValueError(
                "No chunks were created from the PDF."
            )

        storage_path = f"documents/{filename}"

        document_record = self.store.create_document(
            filename=filename,
            storage_path=storage_path
        )

        document_id = document_record["id"]

        embeddings = []

        for chunk in chunks:
            embedding = get_embedding(
                self.client,
                chunk
            )

            embeddings.append(embedding)

        self.store.add_chunks(
            document_id,
            chunks,
            embeddings
        )

        return {
            "filename": filename,
            "document_id": document_id,
            "chunks": len(chunks)
        }
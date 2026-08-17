import fitz

from rag.chunking import chunk_text
from rag.embeddings import get_embedding


class DocumentService:

    def __init__(
        self,
        client,
        store
    ):

        self.client = client
        self.store = store


    # =========================================
    # PROCESS PDF
    # =========================================

    def process_pdf(
        self,
        uploaded_file
    ):

        filename = uploaded_file.name


        # -------------------------------------
        # 1. Read uploaded PDF
        # -------------------------------------

        pdf_bytes = uploaded_file.getvalue()

        if not pdf_bytes:

            raise ValueError(
                "Uploaded PDF is empty."
            )


        # -------------------------------------
        # 2. Extract text
        # -------------------------------------

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


        print(
            f"Extracted {len(text)} characters."
        )


        # -------------------------------------
        # 3. Create chunks
        # -------------------------------------

        chunks = chunk_text(
            text,
            chunk_size=200,
            overlap=50
        )


        print(
            f"Created {len(chunks)} chunks."
        )


        if not chunks:

            raise ValueError(
                "No chunks were created from the PDF."
            )


        # -------------------------------------
        # 4. Create storage path
        # -------------------------------------

        storage_path = (
            f"documents/{filename}"
        )


        # -------------------------------------
        # 5. Create document record
        # -------------------------------------

        document_record = (
            self.store.create_document(
                filename=filename,
                storage_path=storage_path
            )
        )


        document_id = document_record["id"]


        # -------------------------------------
        # 6. Create embeddings
        # -------------------------------------

        embeddings = []


        for i, chunk in enumerate(chunks):

            print(
                f"Creating embedding "
                f"{i + 1}/{len(chunks)}..."
            )


            embedding = get_embedding(
                self.client,
                chunk
            )


            embeddings.append(
                embedding
            )


        # -------------------------------------
        # 7. Store chunks + embeddings
        # -------------------------------------

        self.store.add_chunks(
            document_id,
            chunks,
            embeddings
        )


        print(
            "Document stored successfully."
        )


        # -------------------------------------
        # 8. Return result
        # -------------------------------------

        return {
            "filename": filename,
            "document_id": document_id,
            "chunks": len(chunks)
        }
import os

from dotenv import load_dotenv
from supabase import create_client


load_dotenv()


class SupabaseVectorStore:

    def __init__(self):

        url = os.getenv(
            "SUPABASE_URL"
        )

        key = os.getenv(
            "SUPABASE_SERVICE_ROLE_KEY"
        )

        if not url:
            raise ValueError(
                "SUPABASE_URL is missing."
            )

        if not key:
            raise ValueError(
                "SUPABASE_SERVICE_ROLE_KEY is missing."
            )

        self.client = create_client(
            url,
            key
        )


    # =====================================
    # DOCUMENT
    # =====================================

    def create_document(
        self,
        filename,
        storage_path
    ):

        response = (
            self.client
            .table("documents")
            .insert(
                {
                    "filename": filename,
                    "storage_path": storage_path
                }
            )
            .execute()
        )

        return response.data[0]


    def get_documents(self):

        response = (
            self.client
            .table("documents")
            .select("*")
            .order(
                "created_at",
                desc=True
            )
            .execute()
        )

        return response.data


    def delete_document(
        self,
        document_id
    ):

        self.client.table(
            "documents"
        ).delete().eq(
            "id",
            document_id
        ).execute()


    # =====================================
    # CHUNKS
    # =====================================

    def add_chunks(
        self,
        document_id,
        chunks,
        embeddings
    ):

        records = []

        for chunk_id, (
            chunk,
            embedding
        ) in enumerate(
            zip(
                chunks,
                embeddings
            )
        ):

            records.append(
                {
                    "document_id": document_id,

                    "chunk_id": chunk_id,

                    "content": chunk,

                    "embedding": embedding
                }
            )


        if not records:
            return


        self.client.table(
            "document_chunks"
        ).insert(
            records
        ).execute()


    def search(
        self,
        query_embedding,
        top_k=3
    ):

        response = self.client.rpc(
            "match_document_chunks",
            {
                "query_embedding":
                    query_embedding,

                "match_count":
                    top_k
            }
        ).execute()


        results = []


        for result in response.data:

            results.append(
                {
                    "document_id":
                        result[
                            "document_id"
                        ],

                    "chunk_id":
                        result[
                            "chunk_id"
                        ],

                    "chunk":
                        result[
                            "content"
                        ],

                    "score":
                        float(
                            result[
                                "similarity"
                            ]
                        )
                }
            )


        return results


    def get_all_chunks(self):

        response = (
            self.client
            .table("document_chunks")
            .select(
                "document_id, chunk_id, content"
            )
            .order(
                "chunk_id"
            )
            .execute()
        )

        return response.data


    def get_chunks_for_keyword_search(self):

        return self.get_all_chunks()
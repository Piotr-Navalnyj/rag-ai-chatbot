import re


class KeywordRetriever:

    def __init__(
        self,
        chunks
    ):

        self.chunks = chunks


    def tokenize(
        self,
        text
    ):

        return set(
            re.findall(
                r"\b\w+\b",
                text.lower()
            )
        )


    def search(
        self,
        query,
        top_k=3
    ):

        query_tokens = self.tokenize(
            query
        )

        results = []


        for item in self.chunks:

            chunk = item["content"]

            chunk_tokens = self.tokenize(
                chunk
            )

            if not query_tokens:
                score = 0.0

            else:

                matches = (
                    query_tokens
                    & chunk_tokens
                )

                score = (
                    len(matches)
                    / len(query_tokens)
                )


            results.append(
                {
                    "document_id":
                        item["document_id"],

                    "chunk_id":
                        item["chunk_id"],

                    "chunk":
                        chunk,

                    "score":
                        float(score)
                }
            )


        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )


        return results[:top_k]
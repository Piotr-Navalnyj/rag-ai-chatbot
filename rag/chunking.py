def chunk_text(text, chunk_size=50, overlap=10):

    words = text.split()

    chunks = []

    current_words = []
    current_length = 0

    for word in words:

        word_length = len(word) + 1

        if (
            current_length + word_length
            > chunk_size
            and current_words
        ):

            chunk = " ".join(current_words)

            chunks.append(chunk)

            # Keep the last words for overlap
            overlap_words = []

            overlap_length = 0

            for previous_word in reversed(current_words):

                word_length = len(previous_word) + 1

                if overlap_length + word_length > overlap:
                    break

                overlap_words.insert(
                    0,
                    previous_word
                )

                overlap_length += word_length

            current_words = overlap_words

            current_length = overlap_length

        current_words.append(word)

        current_length += word_length

    if current_words:

        chunks.append(
            " ".join(current_words)
        )

    return chunks
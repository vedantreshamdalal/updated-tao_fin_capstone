
from typing import List, Dict


def split_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 75,
    metadata: Dict | None = None
) -> List[Dict]:
    """
    Split a document into overlapping chunks.

    Each chunk contains:
    - text
    - chunk metadata
    """

    if not text or not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    words = text.split()

    chunks = []

    start = 0
    chunk_id = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )

        chunk_text = " ".join(
            words[start:end]
        )

        chunk_metadata = {
            "chunk_id": chunk_id,
            "start_word": start,
            "end_word": end
        }

        if metadata:
            chunk_metadata.update(metadata)

        chunks.append({
            "text": chunk_text,
            "metadata": chunk_metadata
        })

        chunk_id += 1

        if end >= len(words):
            break

        start = end - overlap

    return chunks
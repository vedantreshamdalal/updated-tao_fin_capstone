
from pathlib import Path

import faiss
import numpy as np


class FAISSVectorStore:

    def __init__(self, dimension: int):
        """
        Create a FAISS index using inner-product similarity.

        Our embeddings are normalized, so inner product
        is equivalent to cosine similarity.
        """

        self.dimension = dimension

        self.index = faiss.IndexFlatIP(
            dimension
        )

        # Keep the original chunks separately.
        self.documents = []

    def add(
        self,
        embeddings,
        documents
    ):
        """
        Add embeddings and their corresponding
        documents to the FAISS index.
        """

        vectors = np.asarray(
            embeddings,
            dtype="float32"
        )

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.dimension}, "
                f"got {vectors.shape[1]}"
            )

        self.index.add(vectors)

        self.documents.extend(
            documents
        )

    def search(
        self,
        query_embedding,
        top_k: int = 5
    ):
        """
        Search for the most similar documents.
        """

        query_vector = np.asarray(
            query_embedding,
            dtype="float32"
        )

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(
                1, -1
            )

        scores, indices = self.index.search(
            query_vector,
            min(top_k, len(self.documents))
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            results.append({
                "score": float(score),
                "index": int(index),
                "document": self.documents[index]
            })

        return results
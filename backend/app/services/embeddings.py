
from sentence_transformers import SentenceTransformer


class EmbeddingService:

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5"
    ):
        self.model_name = model_name

        print(
            f"Loading embedding model: {model_name}"
        )

        self.model = SentenceTransformer(
            model_name
        )

    def embed_documents(self, texts: list[str]):
        """
        Convert multiple documents into vectors.
        """

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def embed_query(self, query: str):
        """
        Convert a user query into a vector.
        """

        return self.model.encode(
            query,
            normalize_embeddings=True
        )

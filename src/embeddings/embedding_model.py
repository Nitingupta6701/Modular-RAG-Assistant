
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL


class EmbeddingModel:
    """
    Generate vector embeddings for text using SentenceTransformer.
    """

    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}")

        self.model = SentenceTransformer(EMBEDDING_MODEL)

        print("Embedding model loaded successfully!")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple text chunks.
        """

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Generate embedding for a single user query.
        """

        if not text.strip():
            raise ValueError("Query cannot be empty.")

        embedding = self.model.encode(
            text,
            convert_to_numpy=True
        )

        return embedding.tolist()

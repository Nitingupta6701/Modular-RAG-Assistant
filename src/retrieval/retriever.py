
from config.settings import TOP_K, MIN_RELEVANCE_SCORE


class Retriever:
    """
    Retrieve relevant document chunks from ChromaDB.
    """

    def __init__(self, vector_store, embedding_model):

        self.vector_store = vector_store
        self.embedding_model = embedding_model

        print("Retriever initialized successfully!")

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        min_score: float = MIN_RELEVANCE_SCORE
    ) -> list[dict]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        # Check whether the database contains documents
        if self.vector_store.get_collection_count() == 0:
            return []

        # Convert user query into an embedding
        query_embedding = self.embedding_model.embed_query(query)

        # Search ChromaDB for similar vectors
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_documents = []

        # Process search results
        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):

            # Convert cosine distance into similarity score
            similarity_score = 1 - distance

            # Apply minimum relevance threshold
            if similarity_score >= min_score:

                retrieved_documents.append({
                    "page_content": document,
                    "metadata": metadata,
                    "score": similarity_score
                })

        return retrieved_documents

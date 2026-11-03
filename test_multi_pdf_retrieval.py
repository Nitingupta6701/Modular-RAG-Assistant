from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.chroma_store import ChromaStore
from src.retrieval.retriever import Retriever


# Step 1: Initialize embedding model
print("\nLoading Embedding Model...")
embedding_model = EmbeddingModel()


# Step 2: Connect to ChromaDB
print("\nConnecting to ChromaDB...")
vector_store = ChromaStore()


# Step 3: Initialize Retriever
print("\nInitializing Retriever...")
retriever = Retriever(
    vector_store=vector_store,
    embedding_model=embedding_model
)


# Step 4: Ask a question
query = input("\nEnter your question: ")

results = retriever.retrieve(
    query=query,
    top_k=5,
    min_score=0.20
)


# Step 5: Display retrieved results
print("\n" + "=" * 50)
print("RETRIEVAL RESULTS")
print("=" * 50)

if not results:
    print("No relevant documents found.")

else:
    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")

        print("Source:", result["metadata"]["file_name"])
        print("Page:", result["metadata"]["page"])
        print("Similarity Score:", round(result["score"], 4))

        print("\nContent:")
        print(result["page_content"][:500])

        print("-" * 50)
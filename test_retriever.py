
from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.chroma_store import ChromaStore
from src.retrieval.retriever import Retriever

# Step 1: Initialize embedding model
print("\nLoading embedding model...")
embedding_model = EmbeddingModel()

# Step 2: Connect to ChromaDB
print("\nConnecting to ChromaDB...")
vector_store = ChromaStore()

# Step 3: Initialize retriever
retriever = Retriever(
    vector_store=vector_store,
    embedding_model=embedding_model
)

# Step 4: Ask a question
query = "Define TF-IDF, BoW, One-hot encoding, NER and Word Sense Disambiguation."

print("\nUser Question:", query)
print("\nSearching relevant documents...")

# Step 5: Retrieve relevant chunks
results = retriever.retrieve(
    query=query,
    top_k=5,
    min_score=0.20
)

# Step 6: Display results
print("\nRetrieval Results")
print("Total relevant chunks:", len(results))

for index, result in enumerate(results, start=1):

    print(f"\n--- Result {index} ---")

    print("Source:", result["metadata"]["file_name"])
    print("Page:", result["metadata"]["page"])
    print("Similarity Score:", round(result["score"], 4))

    print("\nRetrieved Text:")
    print(result["page_content"][:500])

    print("-" * 60)

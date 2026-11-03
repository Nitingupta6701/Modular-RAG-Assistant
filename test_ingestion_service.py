
from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.chroma_store import ChromaStore
from src.ingestion.ingestion_service import IngestionService


# Step 1: Initialize embedding model
print("\nInitializing Embedding Model...")

embedding_model = EmbeddingModel()


# Step 2: Connect to ChromaDB
print("\nConnecting to ChromaDB...")

vector_store = ChromaStore()


# Step 3: Initialize ingestion service
print("\nInitializing Ingestion Service...")

ingestion_service = IngestionService(
    embedding_model=embedding_model,
    vector_store=vector_store
)


# Step 4: Process all PDFs in uploads folder
print("\nStarting PDF ingestion...")

results = ingestion_service.ingest_directory(
    "data/uploads"
)


# Step 5: Display results
print("\n" + "=" * 50)
print("INGESTION SUMMARY")
print("=" * 50)

for result in results:

    print("\nFile:", result["file_name"])
    print("Status:", result["status"])

    if result["status"] == "success":
        print("Pages:", result["pages_processed"])
        print("Chunks:", result["chunks_created"])

    else:
        print("Error:", result["error"])


# Step 6: Display database count
print("\nTotal chunks in ChromaDB:")
print(vector_store.get_collection_count())

print("\nIngestion test completed!")


from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunking import split_documents
from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.chroma_store import ChromaStore

# Step 1: Load PDF
print("\nLoading PDF...")
documents = load_pdf("data/uploads/sample.pdf")

# Step 2: Generate chunks
print("\nGenerating chunks...")
chunks = split_documents(documents)

print("Total chunks:", len(chunks))

# Step 3: Generate embeddings
print("\nLoading embedding model...")
embedding_model = EmbeddingModel()

texts = [chunk["page_content"] for chunk in chunks]
embeddings = embedding_model.embed_documents(texts)

print("Embeddings generated:", len(embeddings))

# Step 4: Initialize ChromaDB
print("\nInitializing ChromaDB...")
vector_store = ChromaStore()

# Step 5: Store chunks and embeddings
print("\nStoring documents...")
vector_store.add_documents(chunks, embeddings)

# Step 6: Verify storage
print("\nVerifying database...")

total_chunks = vector_store.get_collection_count()

print("Total chunks stored in ChromaDB:", total_chunks)


from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunking import split_documents
from src.embeddings.embedding_model import EmbeddingModel

# Step 1: Load PDF
print("\nLoading PDF...")
documents = load_pdf("data/uploads/sample.pdf")

# Step 2: Generate chunks
print("\nGenerating chunks...")
chunks = split_documents(documents)

print("Total chunks:", len(chunks))

# Step 3: Load embedding model
print("\nInitializing embedding model...")
embedding_model = EmbeddingModel()

# Step 4: Extract text from chunks
texts = [chunk["page_content"] for chunk in chunks]

# Step 5: Generate embeddings
print("\nGenerating embeddings...")
embeddings = embedding_model.embed_documents(texts)

# Step 6: Display results
print("\nEmbedding generation successful!")

print("Total embeddings:", len(embeddings))
print("Dimensions per embedding:", len(embeddings[0]))

print("\nFirst embedding (first 10 values):")
print(embeddings[0][:10])

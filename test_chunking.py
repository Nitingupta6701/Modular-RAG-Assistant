
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunking import split_documents

# Step 1: Load PDF
pdf_path = "data/uploads/sample.pdf"

print("Loading PDF...")
documents = load_pdf(pdf_path)

print("Total pages loaded:", len(documents))

# Step 2: Split PDF text into chunks
print("\nSplitting text into chunks...")

chunks = split_documents(documents)

# Step 3: Display results
print("\nChunking completed successfully!")
print("Total chunks generated:", len(chunks))

# Step 4: Display individual chunks
for index, chunk in enumerate(chunks, start=1):

    print(f"\n--- Chunk {index} ---")
    print("Source:", chunk["metadata"]["file_name"])
    print("Page:", chunk["metadata"]["page"])
    print("Characters:", len(chunk["page_content"]))
    print("Content:")
    print(chunk["page_content"][:300])

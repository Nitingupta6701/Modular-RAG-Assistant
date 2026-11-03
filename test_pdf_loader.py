
from src.ingestion.pdf_loader import load_pdf

# Path of our sample PDF
pdf_path = "data/uploads/sample.pdf"

print("Loading PDF...")

try:
    documents = load_pdf(pdf_path)

    print("\nPDF loaded successfully!")
    print("Total pages extracted:", len(documents))

    for document in documents:
        print("\nPage:", document["metadata"]["page"])
        print("File:", document["metadata"]["file_name"])
        print("Extracted text:")
        print(document["page_content"][:500])
        print("-" * 50)

except Exception as error:
    print("\nError:", error)


from pathlib import Path

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunking import split_documents


class IngestionService:
    """
    Manage ingestion of multiple PDF documents
    into ChromaDB.
    """

    def __init__(self, embedding_model, vector_store):

        self.embedding_model = embedding_model
        self.vector_store = vector_store

        print("Ingestion Service initialized successfully!")

    def ingest_pdf(self, file_path: str | Path) -> dict:
        """
        Load, chunk, embed and store a single PDF.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Not a valid file: {path}"
            )

        if path.suffix.lower() != ".pdf":
            raise ValueError(
                "Only PDF files are supported."
            )

        # Use an absolute, normalized path
        path = path.resolve()

        print(f"\nProcessing: {path.name}")

        # Step 1: Extract PDF text
        documents = load_pdf(path)

        # Step 2: Split extracted text into chunks
        chunks = split_documents(documents)

        if not chunks:
            raise ValueError(
                f"No text chunks generated from {path.name}."
            )

        # Step 3: Generate embeddings
        texts = [
            chunk["page_content"]
            for chunk in chunks
        ]

        embeddings = self.embedding_model.embed_documents(
            texts
        )

        # Step 4: Remove previously stored chunks
        # belonging to this exact PDF path
        self.vector_store.delete_by_source(
            str(path)
        )

        # Step 5: Store updated chunks
        self.vector_store.add_documents(
            chunks,
            embeddings
        )

        return {
            "file_name": path.name,
            "source": str(path),
            "pages_processed": len(documents),
            "chunks_created": len(chunks),
            "status": "success"
        }

    def ingest_directory(
        self,
        directory_path: str | Path
    ) -> list[dict]:
        """
        Process all PDF files inside a directory.
        """

        directory = Path(directory_path)

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        if not directory.is_dir():
            raise NotADirectoryError(
                f"Not a directory: {directory}"
            )

        # Resolve directory to an absolute path
        directory = directory.resolve()

        # Find PDF files (case-insensitive extension)
        pdf_files = sorted(
            file for file in directory.iterdir()
            if file.is_file()
            and file.suffix.lower() == ".pdf"
        )

        if not pdf_files:
            raise ValueError(
                "No PDF files found in directory."
            )

        results = []

        # Process each PDF independently
        for pdf_file in pdf_files:

            try:
                result = self.ingest_pdf(pdf_file)
                results.append(result)

            except Exception as error:

                results.append({
                    "file_name": pdf_file.name,
                    "source": str(pdf_file.resolve()),
                    "status": "failed",
                    "error": str(error)
                })

                print(
                    f"Failed to process {pdf_file.name}: {error}"
                )

        return results

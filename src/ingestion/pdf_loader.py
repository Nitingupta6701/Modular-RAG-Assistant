
from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path: str | Path) -> list[dict]:
    """
    Load a PDF and extract text page by page.
    Preserve document information as metadata.
    """

    path = Path(file_path)

    # Check whether the file exists
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    # Check whether the file is a PDF
    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    try:
        reader = PdfReader(str(path))

        documents = []

        # Extract text from each page
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""

            # Skip pages that contain no extractable text
            if page_text.strip():
                documents.append({
                    "page_content": page_text.strip(),
                    "metadata": {
                        "source": str(path.resolve()),
                        "file_name": path.name,
                        "page": page_number
                    }
                })

        if not documents:
            raise ValueError(
                "No extractable text found in the PDF."
            )

        return documents

    except Exception as error:
        if isinstance(error, ValueError):
            raise

        raise RuntimeError(
            f"Failed to read PDF: {error}"
        ) from error

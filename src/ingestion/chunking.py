
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents: list[dict]) -> list[dict]:
    """
    Split extracted PDF documents into smaller chunks
    while preserving their metadata.
    """

    # Configure the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []

    # Process each extracted page
    for document in documents:

        page_text = document["page_content"]
        metadata = document["metadata"]

        # Split page text into smaller pieces
        text_chunks = text_splitter.split_text(page_text)

        # Store each chunk with its metadata
        for chunk in text_chunks:
            chunks.append({
                "page_content": chunk,
                "metadata": metadata.copy()
            })

    return chunks

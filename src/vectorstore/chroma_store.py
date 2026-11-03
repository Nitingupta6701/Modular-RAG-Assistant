
import chromadb

from config.settings import CHROMA_DIR


class ChromaStore:
    """
    Store and manage document embeddings using ChromaDB.
    """

    def __init__(self, collection_name="rag_documents"):

        # Create persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        # Create or access collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print("ChromaDB initialized successfully!")

    def add_documents(self, chunks, embeddings):

        if not chunks:
            raise ValueError("No document chunks provided.")

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        ids = []
        texts = []
        metadatas = []

        # Prepare data for ChromaDB
        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):

            ids.append(
                f"{chunk['metadata']['file_name']}_"
                f"{chunk['metadata']['page']}_{index}"
            )

            texts.append(chunk["page_content"])
            metadatas.append(chunk["metadata"])

        # Store documents and embeddings
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"Successfully stored {len(ids)} chunks!")

    def get_collection_count(self):
        """
        Return the total number of stored chunks.
        """

        return self.collection.count()

    def delete_by_filename(self, file_name: str):
        """
        Delete all stored chunks belonging to a file.
        """

        if not file_name.strip():
            raise ValueError("File name cannot be empty.")

        self.collection.delete(
            where={"file_name": file_name}
        )

        print(f"Deleted existing chunks for: {file_name}")


import chromadb
import hashlib

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

            metadata = chunk["metadata"]

            # Use the full source path to identify a document
            source = metadata.get("source")

            if not source:
                raise ValueError(
                    "Document metadata must contain a source path."
                )

            # Generate a unique document identifier
            source_hash = hashlib.sha256(
                source.encode("utf-8")
            ).hexdigest()[:16]

            # Create a deterministic chunk ID
            chunk_id = (
                f"{source_hash}_"
                f"{metadata.get('page', 0)}_"
                f"{index}"
            )

            ids.append(chunk_id)
            texts.append(chunk["page_content"])
            metadatas.append(metadata.copy())

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

    def delete_by_source(self, source_path: str):
        """
        Delete all chunks belonging to an exact document path.
        """

        if not source_path or not source_path.strip():
            raise ValueError("Source path cannot be empty.")

        self.collection.delete(
            where={"source": source_path}
        )

        print(f"Deleted existing chunks for: {source_path}")

    def delete_by_filename(self, file_name: str):
        """
        Delete chunks belonging to a filename.

        Note: If multiple documents have the same filename,
        this method deletes chunks from all of them.
        Use delete_by_source() for precise deletion.
        """

        if not file_name or not file_name.strip():
            raise ValueError("File name cannot be empty.")

        self.collection.delete(
            where={"file_name": file_name}
        )

        print(f"Deleted existing chunks for: {file_name}")
    
    def get_document_summary(self):
        """
        Return a summary of all indexed PDF documents.
        """

        results = self.collection.get(
            include=["metadatas"]
        )

        documents = {}

        for metadata in results["metadatas"]:

            source = metadata.get("source")

            if not source:
                continue

            if source not in documents:

                documents[source] = {
                    "file_name": metadata.get(
                        "file_name", "Unknown"
                    ),
                    "source": source,
                    "pages": set(),
                    "total_chunks": 0
                }

            page = metadata.get("page")

            if page is not None:
                documents[source]["pages"].add(page)

            documents[source]["total_chunks"] += 1

        summaries = []

        for document in documents.values():

            document["total_pages"] = len(
                document["pages"]
            )

            document.pop("pages")

            summaries.append(document)

        return summaries
    
    def delete_document(self, source_path: str) -> int:
        """
        Delete all chunks belonging to one exact document source.
        Returns the number of deleted chunks.
        """

        if not source_path or not source_path.strip():
            raise ValueError("Source path cannot be empty.")

        # Find chunks belonging to this exact source
        results = self.collection.get(
            where={"source": source_path},
            include=[]
        )

        document_ids = results["ids"]

        if not document_ids:
            return 0

        # Delete only the matching document chunks
        self.collection.delete(ids=document_ids)

        return len(document_ids)

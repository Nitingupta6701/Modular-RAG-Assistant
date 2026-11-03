
from src.vectorstore.chroma_store import ChromaStore

# Step 1: Connect to ChromaDB
print("\nConnecting to ChromaDB...")

store = ChromaStore()

# Step 2: Insert temporary test documents
test_ids = ["delete_test_1", "delete_test_2"]

store.collection.upsert(
    ids=test_ids,
    documents=[
        "Temporary test document one.",
        "Temporary test document two."
    ],
    embeddings=[
        [0.0] * 384,
        [0.1] * 384
    ],
    metadatas=[
        {"file_name": "delete_test.pdf", "page": 1},
        {"file_name": "delete_test.pdf", "page": 2}
    ]
)

# Step 3: Check documents before deletion
before = store.collection.get(
    where={"file_name": "delete_test.pdf"}
)

print("\nChunks before deletion:", len(before["ids"]))

# Step 4: Delete test document chunks
store.delete_by_filename("delete_test.pdf")

# Step 5: Verify deletion
after = store.collection.get(
    where={"file_name": "delete_test.pdf"}
)

print("Chunks after deletion:", len(after["ids"]))

# Step 6: Display result
if len(after["ids"]) == 0:
    print("\nDelete functionality working successfully!")
else:
    print("\nDeletion test failed!")

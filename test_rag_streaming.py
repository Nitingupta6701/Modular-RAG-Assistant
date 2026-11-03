
from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.chroma_store import ChromaStore
from src.retrieval.retriever import Retriever
from src.retrieval.query_rewriter import QueryRewriter
from src.generation.llm import LLMGenerator
from src.pipelines.rag_pipeline import RAGPipeline


# Step 1: Initialize components
print("\nInitializing RAG components...")

embedding_model = EmbeddingModel()
vector_store = ChromaStore()

retriever = Retriever(
    vector_store=vector_store,
    embedding_model=embedding_model
)

query_rewriter = QueryRewriter()
llm_generator = LLMGenerator()


# Step 2: Initialize RAG pipeline
rag = RAGPipeline(
    retriever=retriever,
    query_rewriter=query_rewriter,
    llm_generator=llm_generator
)


# Step 3: Ask a question
question = "What is LCM and how is it calculated?"

print("\nQuestion:", question)
print("\nStreaming RAG Answer:\n")


# Step 4: Generate streaming response
for chunk in rag.ask_stream(question):
    print(chunk, end="", flush=True)


# Step 5: Check conversation memory
print("\n\nConversation turns saved:",
      rag.conversation_memory.get_turn_count())


# Step 6: Display source metadata
metadata = rag.last_stream_metadata

print("\n\n========== STREAMING METADATA ==========")

print("\nOriginal Question:")
print(metadata.get("question"))

print("\nRewritten Query:")
print(metadata.get("rewritten_query"))

print("\nRetrieved Chunks:")
print(metadata.get("retrieved_chunks"))

print("\nSources:")

sources = metadata.get("sources", [])

if sources:

    for index, source in enumerate(sources, start=1):

        print(f"\nSource {index}")
        print("Filename:", source.get("file_name"))
        print("Page Number:", source.get("page"))
        print("Similarity Score:", source.get("score"))

else:
    print("No sources found.")


# Step 7: Verify metadata
assert metadata.get("question") == question
assert "answer" in metadata
assert isinstance(sources, list)

if metadata.get("retrieved_chunks", 0) > 0:

    assert len(sources) > 0

    for source in sources:
        assert "file_name" in source
        assert "page" in source
        assert "score" in source

    print("\nSource metadata verification: PASSED")

else:
    print("\nNo documents retrieved. Source verification skipped.")


print("\nRAG streaming test completed!")

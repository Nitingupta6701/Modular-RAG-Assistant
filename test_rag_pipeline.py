
from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.chroma_store import ChromaStore
from src.retrieval.retriever import Retriever
from src.retrieval.query_rewriter import QueryRewriter
from src.generation.llm import LLMGenerator
from src.pipelines.rag_pipeline import RAGPipeline


# Step 1: Initialize embedding model
print("\nLoading embedding model...")
embedding_model = EmbeddingModel()


# Step 2: Connect to ChromaDB
print("\nConnecting to ChromaDB...")
vector_store = ChromaStore()


# Step 3: Initialize Retriever
print("\nInitializing Retriever...")

retriever = Retriever(
    vector_store=vector_store,
    embedding_model=embedding_model
)


# Step 4: Initialize Query Rewriter
print("\nInitializing Query Rewriter...")
query_rewriter = QueryRewriter()


# Step 5: Initialize Gemini
print("\nInitializing Gemini...")
llm_generator = LLMGenerator()


# Step 6: Initialize RAG Pipeline
print("\nInitializing RAG Pipeline...")

rag = RAGPipeline(
    retriever=retriever,
    query_rewriter=query_rewriter,
    llm_generator=llm_generator
)


# Step 7: Ask a question
question = "What does the document say about TF-IDF?"

print("\nUser Question:", question)
print("\nRunning complete RAG pipeline...")


# Step 8: Generate answer
result = rag.ask(question)


# Step 9: Display results
print("\n" + "=" * 60)
print("RAG PIPELINE RESULT")
print("=" * 60)

print("\nOriginal Question:")
print(result["question"])

print("\nRewritten Query:")
print(result["rewritten_query"])

print("\nGenerated Answer:")
print(result["answer"])

print("\nRetrieved Chunks:")
print(result["retrieved_chunks"])

print("\nSources:")

for source in result["sources"]:
    print(
        f"File: {source['file_name']} | "
        f"Page: {source['page']} | "
        f"Score: {source['score']}"
    )

print("\n" + "=" * 60)

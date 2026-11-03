from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.chroma_store import ChromaStore
from src.retrieval.retriever import Retriever
from src.retrieval.query_rewriter import QueryRewriter
from src.generation.llm import LLMGenerator
from src.pipelines.rag_pipeline import RAGPipeline


print("\nInitializing Embedding Model...")
embedding_model = EmbeddingModel()

print("\nConnecting to ChromaDB...")
vector_store = ChromaStore()

print("\nInitializing Retriever...")
retriever = Retriever(vector_store, embedding_model)

print("\nInitializing Query Rewriter...")
query_rewriter = QueryRewriter()

print("\nInitializing Gemini...")
llm_generator = LLMGenerator()

print("\nInitializing RAG Pipeline...")
rag_pipeline = RAGPipeline(
    retriever=retriever,
    query_rewriter=query_rewriter,
    llm_generator=llm_generator
)

question = input("\nEnter your question: ")

result = rag_pipeline.ask(question)

print("\n" + "=" * 60)
print("RAG RESPONSE")
print("=" * 60)

print("\nQuestion:", result["question"])
print("\nRewritten Query:", result["rewritten_query"])
print("\nAnswer:", result["answer"])

print("\nSources:")

for source in result["sources"]:
    print(
        f"File: {source['file_name']} | "
        f"Page: {source['page']} | "
        f"Score: {source['score']}"
    )

print("\nRetrieved Chunks:", result["retrieved_chunks"])
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

print("\nRAG streaming test completed!")
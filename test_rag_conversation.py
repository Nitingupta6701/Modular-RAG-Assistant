
from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.chroma_store import ChromaStore
from src.retrieval.retriever import Retriever
from src.retrieval.query_rewriter import QueryRewriter
from src.generation.llm import LLMGenerator
from src.pipelines.rag_pipeline import RAGPipeline
from src.memory.conversation import ConversationMemory


# Step 1: Initialize components
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

# Step 2: Initialize conversation memory
print("\nInitializing Conversation Memory...")
memory = ConversationMemory(max_turns=5)

# Step 3: Initialize RAG Pipeline
rag_pipeline = RAGPipeline(
    retriever=retriever,
    query_rewriter=query_rewriter,
    llm_generator=llm_generator,
    conversation_memory=memory
)

# Step 4: Ask first question
question1 = input("\nQuestion 1: ")

result1 = rag_pipeline.ask(question1)

print("\nAnswer 1:")
print(result1["answer"])

# Step 5: Ask follow-up question
question2 = input("\nQuestion 2 (Follow-up): ")

result2 = rag_pipeline.ask(question2)

print("\nAnswer 2:")
print(result2["answer"])

# Step 6: Display conversation history
print("\n" + "=" * 50)
print("SAVED CONVERSATION HISTORY")
print("=" * 50)

print(memory.get_history())

print("\nTotal Conversation Turns:")
print(memory.get_turn_count())

print("\nConversation Memory Integration Test Completed!")

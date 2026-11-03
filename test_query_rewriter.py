
from src.retrieval.query_rewriter import QueryRewriter

# Step 1: Initialize Query Rewriter
print("\nInitializing Query Rewriter...")

rewriter = QueryRewriter()

# Step 2: Provide conversation history
chat_history = """
User: What is Natural Language Processing?
Assistant: NLP is a field of Artificial Intelligence
that helps computers understand human language.
"""

# Step 3: Ask a follow-up question
query = "What are its applications?"

print("\nOriginal Question:", query)

# Step 4: Rewrite the question
print("\nRewriting query...")

rewritten_query = rewriter.rewrite_query(
    query=query,
    chat_history=chat_history
)

# Step 5: Display result
print("\nOriginal Query:")
print(query)

print("\nRewritten Query:")
print(rewritten_query)

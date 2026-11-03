
from src.generation.prompt_templates import build_rag_prompt

# Sample retrieved document context
context = """
Source: NLP_Notes.pdf
Page: 2

Natural Language Processing (NLP) is a branch
of Artificial Intelligence that enables computers
to understand and process human language.
"""

# User question
question = "What is Natural Language Processing?"

# Build the prompt
prompt = build_rag_prompt(
    question=question,
    context=context
)

# Display the generated prompt
print("\nRAG Prompt Generated Successfully!")
print("\n" + "=" * 60)
print(prompt)
print("=" * 60)

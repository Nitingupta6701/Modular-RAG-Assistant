
from src.generation.llm import LLMGenerator


# Step 1: Initialize Gemini
print("\nInitializing Gemini...")
llm = LLMGenerator()


# Step 2: Prepare prompt
prompt = """
Explain Natural Language Processing (NLP)
in simple English with three examples.
"""


# Step 3: Generate streaming response
print("\nGemini Streaming Response:\n")

for chunk in llm.generate_stream(prompt):
    print(chunk, end="", flush=True)


print("\n\nStreaming test completed!")

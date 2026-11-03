
from src.generation.llm import LLMGenerator

# Step 1: Initialize Gemini
print("\nInitializing Gemini LLM...")

llm = LLMGenerator()

# Step 2: Create a test prompt
prompt = """
Explain Natural Language Processing in simple words.
Give its three applications with examples.
"""

# Step 3: Generate an answer
print("\nSending prompt to Gemini...")

answer = llm.generate(prompt)

# Step 4: Display the response
print("\nGemini Response:")
print(answer)

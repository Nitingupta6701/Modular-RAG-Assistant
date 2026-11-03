from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import GOOGLE_API_KEY, GEMINI_MODEL

print("Initializing Gemini...")

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    timeout=20,
    max_retries=0
)

print("Sending request to Gemini...")

try:
    response = llm.invoke(
        "Explain RAG in two simple sentences."
    )

    print("\nGemini Response:")
    print(response.content)

except Exception as e:
    print("\nGemini API Error:")
    print(type(e).__name__)
    print(e)
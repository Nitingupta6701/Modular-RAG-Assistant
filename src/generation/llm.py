
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import GOOGLE_API_KEY, GEMINI_MODEL


class LLMGenerator:

    def __init__(self):

        if not GOOGLE_API_KEY:
            raise ValueError("Google API key is missing.")

        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY
        )

        print("Gemini LLM initialized successfully!")

    def generate(self, prompt: str) -> str:

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        response = self.llm.invoke(prompt)

        return response.text.strip()

    def generate_stream(self, prompt: str):

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        for chunk in self.llm.stream(prompt):

            if chunk.text:
                yield chunk.text

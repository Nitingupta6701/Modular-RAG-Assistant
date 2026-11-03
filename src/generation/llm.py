
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import GOOGLE_API_KEY, GEMINI_MODEL


class LLMGenerator:
    """
    Generate answers using Google's Gemini model.
    """

    def __init__(self):

        if not GOOGLE_API_KEY:
            raise ValueError("Google API key is missing.")

        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY
        )

        print("Gemini LLM initialized successfully!")

    def generate(self, prompt: str) -> str:
        """
        Generate a text response from Gemini.
        """

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        response = self.llm.invoke(prompt)

        return response.text.strip()

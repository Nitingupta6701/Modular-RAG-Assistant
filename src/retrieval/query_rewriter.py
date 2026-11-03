
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import GOOGLE_API_KEY, GEMINI_MODEL


class QueryRewriter:
    """
    Rewrite user queries into clear, standalone questions.
    """

    def __init__(self):

        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )

        print("Query Rewriter initialized successfully!")

    def rewrite_query(
        self,
        query: str,
        chat_history: str = ""
    ) -> str:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        prompt = f"""
        You are a query rewriting assistant.

        Your task is to convert the user's question
        into a clear and standalone search query.

        Use conversation history only to understand
        references in the current question.

        Do not answer the question.
        Do not add information that is not provided.

        Conversation History:
        {chat_history}

        User Question:
        {query}

        Return only the rewritten question.
        """

        response = self.llm.invoke(prompt)

        rewritten_query = response.text.strip()

        return rewritten_query

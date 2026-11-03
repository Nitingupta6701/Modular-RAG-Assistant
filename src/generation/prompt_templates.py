
def build_rag_prompt(
    question: str,
    context: str,
    chat_history: str = ""
) -> str:
    """
    Build a prompt that instructs Gemini
    to answer using retrieved document context.
    """

    prompt = f"""
You are a helpful AI assistant that answers
questions using the provided documents.

STRICT INSTRUCTIONS:

1. Answer the question using ONLY the
   information available in the context.

2. Do not invent facts or use outside knowledge.

3. If the context does not contain enough
   information, clearly say:
   "I could not find sufficient information
   in the uploaded documents."

4. Give clear, structured and easy-to-understand
   answers.

5. Use conversation history only to understand
   the user's current question. Do not treat
   previous assistant answers as document evidence.

6. Mention the source filename and page number
   when referring to document information.

CONVERSATION HISTORY:
{chat_history if chat_history else "No previous conversation."}

RETRIEVED DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    return prompt.strip()

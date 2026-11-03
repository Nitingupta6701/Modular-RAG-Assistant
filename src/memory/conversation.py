class ConversationMemory:
    """
    Store and manage conversation history
    between the user and AI assistant.
    """

    def __init__(self, max_turns: int = 5):

        self.max_turns = max_turns
        self.chat_history = []

        print("Conversation Memory initialized successfully!")

    def add_message(self, user_message: str, assistant_message: str):

        if not user_message.strip():
            raise ValueError("User message cannot be empty.")

        if not assistant_message.strip():
            raise ValueError("Assistant message cannot be empty.")

        self.chat_history.append({
            "user": user_message,
            "assistant": assistant_message
        })

        # Keep only the latest conversation turns
        self.chat_history = self.chat_history[-self.max_turns:]

    def get_history(self) -> str:

        history = []

        for conversation in self.chat_history:

            history.append(
                f"User: {conversation['user']}"
            )

            history.append(
                f"Assistant: {conversation['assistant']}"
            )

        return "\n".join(history)

    def clear_history(self):

        self.chat_history.clear()

        print("Conversation history cleared successfully!")

    def get_turn_count(self) -> int:

        return len(self.chat_history)
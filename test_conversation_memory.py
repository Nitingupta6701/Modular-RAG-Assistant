from src.memory.conversation import ConversationMemory


# Step 1: Initialize conversation memory
memory = ConversationMemory(max_turns=5)


# Step 2: Store first conversation
memory.add_message(
    user_message="What is LCM?",
    assistant_message="LCM stands for Least Common Multiple."
)


# Step 3: Store second conversation
memory.add_message(
    user_message="Give me an example.",
    assistant_message="The LCM of 4 and 6 is 12."
)


# Step 4: Display conversation history
print("\nConversation History:")
print(memory.get_history())


# Step 5: Display total conversation turns
print("\nTotal Conversation Turns:")
print(memory.get_turn_count())


# Step 6: Clear conversation history
memory.clear_history()

print("\nHistory after clearing:")
print(memory.get_history())

print("\nConversation Memory Test Completed!")
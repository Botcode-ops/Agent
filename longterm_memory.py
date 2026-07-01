# pyrefly: ignore [missing-import]
from llama_cpp import Llama
from typing import List, Dict, Any
import json
import os
import sys

class LongTermMemoryBot:
    """A chatbot with persistent user memory stored in a local JSON file."""

    def __init__(self, model_path: str = "myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", memory_file: str = "memory.json") -> None:
        self.llm = Llama(model_path=model_path)
        self.memory_file = memory_file
        self.user_name = self._load_or_create_memory()
        self.history: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": f"You are a helpful assistant. The user's name is {self.user_name}. Always be polite to them."
            }
        ]

    def _load_or_create_memory(self) -> str:
        """Loads existing memory or prompts user for name to initialize new memory."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as file:
                    memory_data = json.load(file)
                user_name = memory_data.get("user_name", "Friend")
                print(f"--- [SYSTEM: Memory Loaded! Welcome back, {user_name}] ---")
                return str(user_name)
            except Exception as e:
                print(f"--- [SYSTEM: Error loading memory file, starting fresh: {e}] ---")

        # Create new memory
        print("AI: Hello! I don't believe we've met. What is your name?")
        try:
            user_name = input("You (Your Name): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            sys.exit(0)
        
        if not user_name:
            user_name = "Friend"

        memory_data = {"user_name": user_name}
        try:
            with open(self.memory_file, "w") as file:
                json.dump(memory_data, file)
            print(f"--- [SYSTEM: Memory Saved! Nice to meet you, {user_name}] ---")
        except Exception as e:
            print(f"--- [SYSTEM: Error saving memory file: {e}] ---")

        return user_name

    def send_message(self, message: str) -> str:
        """Send message to assistant and append to conversation history."""
        self.history.append({"role": "user", "content": message})
        response = self.llm.create_chat_completion(messages=self.history)
        ai_text = str(response["choices"][0]["message"]["content"])
        self.history.append({"role": "assistant", "content": ai_text})
        return ai_text

def main() -> None:
    print("Initializing Long Term Memory Bot...")
    try:
        bot = LongTermMemoryBot()
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        sys.exit(1)

    print("\n--- Long Term Memory Chat Loop ---")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:
            user_text = input(f"\n{bot.user_name}: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_text:
            continue

        if user_text.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        ai_response = bot.send_message(user_text)
        print(f"AI: {ai_response}")

if __name__ == "__main__":
    main()

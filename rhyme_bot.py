# pyrefly: ignore [missing-import]
from llama_cpp import Llama
from typing import List, Dict
import sys

class RhymeBot:
    """A chatbot that responds exclusively in rhymes using system prompt constraints."""

    def __init__(self, model_path: str = "myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf") -> None:
        self.llm = Llama(model_path=model_path)
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant that only answers in rhymes."}
        ]

    def send_message(self, message: str) -> str:
        """Send user message and return the rhyming response."""
        self.history.append({"role": "user", "content": message})
        response = self.llm.create_chat_completion(messages=self.history)
        ai_text = str(response["choices"][0]["message"]["content"])
        self.history.append({"role": "assistant", "content": ai_text})
        return ai_text

def main() -> None:
    print("Initializing Rhyme Bot...")
    try:
        bot = RhymeBot()
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        sys.exit(1)

    print("\n--- Rhyme Bot Loop ---")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:
            user_text = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_text:
            continue

        if user_text.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        ai_response = bot.send_message(user_text)
        print(ai_response)

if __name__ == "__main__":
    main()

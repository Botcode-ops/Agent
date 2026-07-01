# pyrefly: ignore [missing-import]
from llama_cpp import Llama
from typing import List, Dict, Any
import sys

class BasicChatbot:
    """A basic interactive chat assistant using a local Llama model."""

    def __init__(self, model_path: str = "myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf") -> None:
        self.llm = Llama(model_path=model_path)
        self.history: List[Dict[str, str]] = []

    def test_prompt(self, prompt: str) -> str:
        """Run a simple one-off prompt completion."""
        response = self.llm(prompt, max_tokens=25)
        return str(response["choices"][0]["text"])

    def send_message(self, message: str) -> str:
        """Send a user message and record the interaction in chat history."""
        self.history.append({"role": "user", "content": message})
        response = self.llm.create_chat_completion(messages=self.history)
        ai_text = str(response["choices"][0]["message"]["content"])
        self.history.append({"role": "assistant", "content": ai_text})
        return ai_text

def main() -> None:
    print("Initializing Basic Chatbot...")
    try:
        bot = BasicChatbot()
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        sys.exit(1)

    print("Initial test prompt: 'the moon made of'")
    print(bot.test_prompt("the moon made of"))

    print("\n--- Basic Chat Loop ---")
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

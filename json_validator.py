# pyrefly: ignore [missing-import]
from llama_cpp import Llama
from typing import List, Dict, Any, Optional
import json
import sys

class JSONValidatorBot:
    """A chatbot that enforces valid JSON outputs with built-in retry mechanics."""

    def __init__(self, model_path: str = "myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf") -> None:
        self.llm = Llama(model_path=model_path)
        self.history: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": "You are a data assistant. You ONLY output valid JSON. Do not include any other text."
            }
        ]

    def send_message_with_retry(self, message: str, max_attempts: int = 3) -> Optional[Dict[str, Any]]:
        """Send message and retry up to max_attempts to ensure the response parses as valid JSON."""
        self.history.append({"role": "user", "content": message})
        
        for i in range(max_attempts):
            print(f"--- Attempt {i+1} ---")
            try:
                response = self.llm.create_chat_completion(messages=self.history)
                ai_text = str(response["choices"][0]["message"]["content"])
                
                # Validation Step
                data = json.loads(ai_text)
                print("Success! Valid JSON found:")
                
                # Save valid response to history and return
                self.history.append({"role": "assistant", "content": ai_text})
                return data
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Error: The AI didn't send valid JSON. ({e})")
                if i == max_attempts - 1:
                    print("Final attempt failed. Please try a different question.")
        
        return None

def main() -> None:
    print("Initializing JSON Validator Bot...")
    try:
        bot = JSONValidatorBot()
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        sys.exit(1)

    print("Assistant: I am ready to give you structured JSON data!")
    print("\n--- JSON Validator Loop ---")
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

        result = bot.send_message_with_retry(user_text)
        if result is not None:
            print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

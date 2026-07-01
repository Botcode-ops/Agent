# pyrefly: ignore [missing-import]
from llama_cpp import Llama
from typing import List, Dict, Any, Tuple
import sys

class RouterAgentBot:
    """A chatbot that routes incoming requests to specialized agents based on classification."""

    def __init__(self, model_path: str = "myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf") -> None:
        self.llm = Llama(model_path=model_path)
        self.classifier_prompt = (
            "You are a routing assistant. Your job is to read the user's message \n"
            "and pick the correct category. Only output the category name.\n\n"
            "CATEGORIES:\n"
            "- MATH: For calculations and numbers.\n"
            "- WEATHER: For questions about rain, sun, or temperature.\n"
            "- GENERAL: For greetings, chat, or anything else.\n\n"
            "Response format: ONLY the category name in uppercase."
        )
        self.math_expert_prompt = "You are a math expert. Provide clear, step-by-step solutions to math problems."

    def classify_query(self, user_query: str) -> str:
        """Classify the user query into MATH, WEATHER, or GENERAL."""
        messages = [
            {"role": "system", "content": self.classifier_prompt},
            {"role": "user", "content": user_query}
        ]
        response = self.llm.create_chat_completion(messages=messages)
        category = str(response["choices"][0]["message"]["content"]).strip().upper()
        return category

    def process_query(self, user_query: str) -> Tuple[str, str]:
        """Classify and route the query, returning the specialist name and response."""
        category = self.classify_query(user_query)
        
        if "MATH" in category:
            messages = [
                {"role": "system", "content": self.math_expert_prompt},
                {"role": "user", "content": user_query}
            ]
            response = self.llm.create_chat_completion(messages=messages)
            return "Math Expert", str(response["choices"][0]["message"]["content"])
        
        elif "WEATHER" in category:
            return "Weather System", "It is currently rainy!"
            
        else:
            return "General Specialist", "Hello! How can I help you today?"

def main() -> None:
    print("Initializing Router Agent Bot...")
    try:
        bot = RouterAgentBot()
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        sys.exit(1)

    print("Assistant: I am ready! I will route your request to the right specialist.")
    print("\n--- Router Agent Loop ---")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_text:
            continue

        if user_text.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        specialist, response = bot.process_query(user_text)
        print(f"--- [RECEPTIONIST: Routing to {specialist.upper()}] ---")
        print(f"AI ({specialist}): {response}")

if __name__ == "__main__":
    main()

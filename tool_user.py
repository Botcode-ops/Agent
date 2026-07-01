# pyrefly: ignore [missing-import]
from llama_cpp import Llama
from typing import List, Dict, Any, Optional
import json
import sys

class ToolUserAgent:
    """An agent that can dynamically choose and execute local Python tools based on system prompts."""

    def __init__(self, model_path: str = "myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf") -> None:
        self.llm = Llama(model_path=model_path)
        self.history: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful assistant with tools."}
        ]
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"}
                        },
                        "required": ["city"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform math like add or multiple",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                            "operation": {"type": "string", "enum": ["add", "multiple"]}
                        },
                        "required": ["a", "b", "operation"]
                    }
                }
            }
        ]

    def get_weather(self, city: str) -> str:
        """Call get_weather tool."""
        print(f"\n[System: Calling get_weather for {city}]")
        return "It is rainy."

    def calculate(self, a: float, b: float, operation: str) -> str:
        """Call calculate tool."""
        print(f"\n[System: Calling calculate for {operation}]")
        op = operation.strip().lower()
        if op == "add":
            return str(a + b)
        elif op == "multiple":
            return str(a * b)
        return "Error: Unknown operation"

    def handle_message(self, message: str) -> str:
        """Send message, execute tool calls if triggered, and return the final response."""
        self.history.append({"role": "user", "content": message})
        
        response = self.llm.create_chat_completion(messages=self.history, tools=self.tools)
        msg_obj = response["choices"][0]["message"]

        if msg_obj.get("tool_calls"):
            self.history.append(msg_obj)
            
            for tool_call in msg_obj["tool_calls"]:
                fn_name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"])
                
                result = ""
                if fn_name == "get_weather":
                    result = self.get_weather(args.get("city", ""))
                elif fn_name == "calculate":
                    result = self.calculate(
                        a=float(args.get("a", 0)), 
                        b=float(args.get("b", 0)), 
                        operation=str(args.get("operation", "add"))
                    )
                
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": fn_name,
                    "content": result
                })

            final_response = self.llm.create_chat_completion(messages=self.history)
            ai_text = str(final_response["choices"][0]["message"]["content"])
            self.history.append({"role": "assistant", "content": ai_text})
            return ai_text
        else:
            ai_text = str(msg_obj.get("content", ""))
            if ai_text:
                self.history.append({"role": "assistant", "content": ai_text})
                return ai_text
            else:
                return "(No response)"

def main() -> None:
    print("Initializing Tool User Agent...")
    try:
        bot = ToolUserAgent()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)

    print("\n--- Tool User Chat Loop ---")
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

        ai_response = bot.handle_message(user_text)
        print(f"AI: {ai_response}")

if __name__ == "__main__":
    main()

# pyrefly: ignore [missing-import]
from llama_cpp import Llama
from typing import List, Dict, Any, Optional
import json
import sys

class AutonomousReActAgent:
    """An autonomous Reason-Act-Observe loop (ReAct) agent that calls functions and reasons iteratively."""

    def __init__(self, model_path: str = "myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf") -> None:
        self.llm = Llama(model_path=model_path)
        self.system_prompt = (
            "You are an autonomous agent. Solve the user's request step-by-step.\n"
            "For each step, you must follow this format:\n\n"
            "THOUGHT: (Explain what you are doing)\n"
            "ACTION: (Call a tool if needed)\n\n"
            "When you have the final result, you MUST say:\n"
            "FINAL ANSWER: (Your final response)"
        )
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform math operations",
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

    def calculate(self, a: float, b: float, operation: str) -> str:
        """Call calculate tool."""
        print(f"\n[System: Executing calculate {a} {operation} {b}]")
        op = operation.strip().lower()
        if op == "add":
            return str(a + b)
        if op == "multiple":
            return str(a * b)
        return "Error"

    def run(self, user_request: str, max_steps: int = 5) -> Optional[str]:
        """Execute the ReAct loop up to max_steps."""
        history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_request}
        ]

        for i in range(max_steps):
            print(f"--- [AI Thinking Step {i+1}] ---")
            
            response = self.llm.create_chat_completion(messages=history, tools=self.tools)
            message = response["choices"][0]["message"]
            
            # Check if the AI is finished
            content = message.get("content")
            if content and "FINAL ANSWER" in content:
                print(f"AI: {content}")
                return str(content)
            
            # Handle tool calls
            if message.get("tool_calls"):
                history.append(message)
                for tool_call in message["tool_calls"]:
                    fn_name = tool_call["function"]["name"]
                    args = json.loads(tool_call["function"]["arguments"])
                    
                    result = self.calculate(
                        a=float(args.get("a", 0)), 
                        b=float(args.get("b", 0)), 
                        operation=str(args.get("operation", "add"))
                    )
                    
                    history.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": fn_name,
                        "content": result
                    })
                    print(f"Observation: Result is {result}")
            else:
                # No tool calls, append message to history
                history.append(message)
                if content:
                    print(f"AI Thought: {content}")
        
        return None

def main() -> None:
    print("Initializing Autonomous ReAct Agent...")
    try:
        bot = AutonomousReActAgent()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)

    print("\n--- Autonomous ReAct Loop ---")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:
            user_text = input("\nUser Request: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_text:
            continue

        if user_text.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        bot.run(user_text)

if __name__ == "__main__":
    main()

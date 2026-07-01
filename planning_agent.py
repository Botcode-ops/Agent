# pyrefly: ignore [missing-import]
from llama_cpp import Llama, LlamaGrammar
from typing import List, Dict, Any, Optional
import json
import sys

class PlanningAgentBot:
    """An agent that creates a 3-step execution plan and resolves each step sequentially using tools."""

    def __init__(self, model_path: str = "myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf") -> None:
        self.llm = Llama(model_path=model_path)
        
        self.planner_system_prompt = (
            "You are a planning assistant. Your job is to take a user request and break it \n"
            "down into exactly 3 logical steps. \n\n"
            "Format your response ONLY as a JSON list of strings.\n"
            "Example: [\"Step 1\", \"Step 2\", \"Step 3\"]"
        )
        
        grammar_text = r"""
        root   ::= "[" ws string "," ws string "," ws string ws "]"
        string ::= "\"" ([^"\\\\] | "\\" ["\\/bfnrt] | "\\" "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])* "\""
        ws     ::= [ \t\n\r]*
        """
        self.planner_grammar = LlamaGrammar.from_string(grammar_text)
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform math like add or multiply",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                            "operation": {"type": "string", "enum": ["add", "multiply"]}
                        },
                        "required": ["a", "b", "operation"]
                    }
                }
            }
        ]

    def calculate(self, a: float, b: float, operation: str) -> str:
        """Perform math calculation tool."""
        print(f"\n[SYSTEM: Executing {operation} on {a} and {b}]")
        op = operation.strip().lower()
        if op == "add":
            return str(a + b)
        if op == "multiply":
            return str(a * b)
        return "0"

    def create_plan(self, user_request: str) -> List[str]:
        """Generate a 3-step execution plan using grammar constraints."""
        plan_messages = [
            {"role": "system", "content": self.planner_system_prompt},
            {"role": "user", "content": f"Create a plan for: {user_request}"}
        ]
        response = self.llm.create_chat_completion(messages=plan_messages, grammar=self.planner_grammar)
        plan_text = str(response["choices"][0]["message"]["content"])
        return list(json.loads(plan_text))

    def execute_plan(self, plan: List[str]) -> str:
        """Execute each step in the plan sequentially and update context."""
        context = ""
        for step in plan:
            print(f"\nCurrent Task: {step}")
            worker_messages = [
                {"role": "system", "content": f"You are a worker. Solve this step: {step}. Context: {context}"},
                {"role": "user", "content": "Use the calculate tool if needed, otherwise provide the answer."}
            ]
            
            step_response = self.llm.create_chat_completion(messages=worker_messages, tools=self.tools)
            message = step_response["choices"][0]["message"]

            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    fn_name = tool_call["function"]["name"]
                    args = json.loads(tool_call["function"]["arguments"])
                    
                    if fn_name == "calculate":
                        result = self.calculate(
                            a=float(args.get("a", 0)), 
                            b=float(args.get("b", 0)), 
                            operation=str(args.get("operation", "add"))
                        )
                        print(f"Tool Result: {result}")
                        context += f"\nResult of '{step}' (via tool): {result}"
            else:
                step_result = str(message.get("content", ""))
                print(f"Task Result: {step_result}")
                context += f"\nResult of '{step}': {step_result}"

        # Final Summary
        print("\n--- FINAL ANSWER ---")
        final_messages = [
            {"role": "system", "content": "Summarize the final result based on these completed steps."},
            {"role": "user", "content": context}
        ]
        summary_response = self.llm.create_chat_completion(messages=final_messages)
        return str(summary_response["choices"][0]["message"]["content"])

def main() -> None:
    print("--- Lesson 8: The Planning Agent ---")
    try:
        bot = PlanningAgentBot()
    except Exception as e:
        print(f"Error initializing planning agent: {e}")
        sys.exit(1)

    while True:
        try:
            user_request = input("\nWhat complex task should I plan? (e.g., 'Add 5 and 10 then multiply by 2'): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_request:
            continue

        if user_request.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            print("\nAI is creating a plan...")
            plan = bot.create_plan(user_request)
            print("--- THE PLAN ---")
            for i, step in enumerate(plan):
                print(f"{i+1}. {step}")
            
            print("\n--- EXECUTING PLAN ---")
            summary = bot.execute_plan(plan)
            print(summary)
        except Exception as e:
            print(f"Error during execution: {e}")

if __name__ == "__main__":
    main()

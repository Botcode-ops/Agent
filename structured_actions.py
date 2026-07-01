# pyrefly: ignore [missing-import]
from llama_cpp import Llama, LlamaGrammar
from typing import List, Dict, Any, Optional
import json
import sys

class StructuredActionsAgent:
    """An autonomous agent that generates and executes structured JSON actions via GBNF grammar constraints."""

    def __init__(self, model_path: str = "myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf") -> None:
        self.llm = Llama(model_path=model_path)
        grammar_text = r"""
        root ::= main
        main ::= "{" ws "\"thought\"" ws ":" ws string "," ws "\"action\"" ws ":" ws act "," ws "\"params\"" ws ":" ws prm ws "}"
        act ::= "\"calculate\"" | "\"final_answer\""
        prm ::= obj | "null"
        obj ::= "{" ws (string ws ":" ws val (ws "," ws string ws ":" ws val)*)? ws "}"
        val ::= obj | arr | string | num | "true" | "false" | "null"
        arr ::= "[" ws (val (ws "," ws val)*)? ws "]"
        string ::= "\"" ([^"\\\\] | "\\" ["\\/bfnrt] | "\\" "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])* "\""
        num ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
        ws ::= [ \t\n\r]*
        """
        self.grammar = LlamaGrammar.from_string(grammar_text)
        self.system_prompt = (
            "You are a structured agent. You MUST respond with JSON containing 'thought', 'action', and 'params'.\n"
            "Available actions:\n"
            "- calculate: a, b, operation ('add', 'multiply')\n"
            "- final_answer: answer\n"
            "Example: {\"thought\": \"I need to add these\", \"action\": \"calculate\", \"params\": {\"a\": 5, \"b\": 3, \"operation\": \"add\"}}"
        )

    def calculate(self, a: Any, b: Any, operation: str) -> str:
        """Execute calculation tool."""
        print(f"\n[SYSTEM: Executing {operation} on {a} and {b}]")
        try:
            num_a = float(a)
            num_b = float(b)
            op = operation.strip().lower()
            if op == "add":
                return str(num_a + num_b)
            if op == "multiply":
                return str(num_a * num_b)
            return f"Unsupported operation: {op}"
        except Exception as e:
            return f"Error: {e}"

    def run_loop(self, user_request: str, max_steps: int = 5) -> Optional[str]:
        """Run the autonomous ReAct structured loop."""
        history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_request}
        ]

        for step in range(max_steps):
            print(f"\n--- [Step {step + 1}] ---")
            
            response = self.llm.create_chat_completion(
                messages=history,
                grammar=self.grammar
            )
            output_text = str(response["choices"][0]["message"]["content"])
            
            try:
                action_obj = json.loads(output_text)
                thought = action_obj.get("thought")
                action = action_obj.get("action")
                params = action_obj.get("params", {})

                print(f"THOUGHT: {thought}")
                print(f"ACTION: {action}({params})")

                if action == "final_answer":
                    answer = params.get("answer")
                    print(f"AI FINAL ANSWER: {answer}")
                    return str(answer)
                
                if action == "calculate":
                    result = self.calculate(params.get("a"), params.get("b"), str(params.get("operation", "")))
                    print(f"OBSERVATION: {result}")
                    
                    history.append({"role": "assistant", "content": output_text})
                    history.append({"role": "user", "content": f"Observation: {result}"})
                else:
                    print(f"SYSTEM: Unknown action '{action}'.")
                    history.append({"role": "assistant", "content": output_text})
                    history.append({"role": "user", "content": "Error: Unknown action. Try again."})
            except Exception as e:
                print(f"ERROR: Failed to parse action. {e}")
                print(f"AI said: {output_text}")
                break
        
        return None

def main() -> None:
    print("--- Lesson 9: Structured Actions (JSON Agent) ---")
    print("Available Actions: 'calculate' (params: a, b, operation), 'final_answer' (params: answer)")
    try:
        bot = StructuredActionsAgent()
    except Exception as e:
        print(f"Error initializing structured agent: {e}")
        sys.exit(1)

    while True:
        try:
            user_request = input("\nUser: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_request:
            continue

        if user_request.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        bot.run_loop(user_request)

if __name__ == "__main__":
    main()

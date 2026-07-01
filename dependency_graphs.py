# pyrefly: ignore [missing-import]
from llama_cpp import Llama, LlamaGrammar
from typing import List, Dict, Any, Set, Tuple
import json
import sys

class DependencyGraphAgent:
    """An agent that breaks complex queries down into a directed acyclic graph (DAG) of tasks and resolves them."""

    def __init__(self, model_path: str = "myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf") -> None:
        self.llm = Llama(model_path=model_path)
        grammar_text = r"""
        root   ::= object
        object ::= "{" ws "\"tasks\":" ws "[" ws task_list ws "]" ws "}"
        task_list ::= task (ws "," ws task)*
        task   ::= "{" ws "\"id\":" ws number "," ws "\"description\":" ws string "," ws "\"depends_on\":" ws "[" ws id_list ws "]" ws "}"
        id_list ::= (number (ws "," ws number)*)?
        string ::= "\"" ([^"\\\\] | "\\" ["\\/bfnrt] | "\\" "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])* "\""
        number ::= [0-9]+
        ws     ::= [ \t\n\r]*
        """
        self.grammar = LlamaGrammar.from_string(grammar_text)

    def calculate(self, operation: str, a: float, b: float) -> float:
        """Perform simple math operation."""
        print(f"\n[SYSTEM: Executing {operation} on {a} and {b}]")
        op = operation.strip().lower()
        if op == "add":
            return a + b
        if op == "multiply":
            return a * b
        return 0.0

    def generate_graph(self, user_request: str) -> List[Dict[str, Any]]:
        """Generate a structured list of tasks and their dependencies using grammar."""
        prompt = (
            f"Break this request into a dependency graph: {user_request}\n"
            "Tasks should be atomic math operations. Use the 'id' to reference dependencies."
        )
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            grammar=self.grammar
        )
        graph_data = json.loads(response["choices"][0]["message"]["content"])
        return list(graph_data.get("tasks", []))

    def execute_graph(self, tasks: List[Dict[str, Any]]) -> Dict[int, Any]:
        """Execute tasks in topological order resolving dependencies sequentially."""
        results: Dict[int, Any] = {}
        completed_ids: Set[int] = set()

        while len(completed_ids) < len(tasks):
            found_work = False
            for t in tasks:
                task_id = int(t["id"])
                if task_id in completed_ids:
                    continue

                depends_on = [int(dep) for dep in t.get("depends_on", [])]
                if all(dep_id in completed_ids for dep_id in depends_on):
                    print(f"\nRunning Task {task_id}: {t['description']}")
                    
                    # Collate context from previous resolved task values
                    context = "\n".join([f"Result of Task {rid}: {res}" for rid, res in results.items() if rid in depends_on])
                    
                    worker_prompt = (
                        f"Solve this task: {t['description']}\n"
                        f"Previous Context: {context}\n"
                        "Provide ONLY the numeric result."
                    )
                    
                    worker_response = self.llm.create_chat_completion(
                        messages=[{"role": "user", "content": worker_prompt}]
                    )
                    res_text = str(worker_response["choices"][0]["message"]["content"]).strip()
                    
                    try:
                        res_val = float(''.join(c for c in res_text if c.isdigit() or c == '.'))
                        results[task_id] = res_val
                        print(f"Result: {res_val}")
                    except ValueError:
                        results[task_id] = res_text
                        print(f"Result: {res_text}")
                        
                    completed_ids.add(task_id)
                    found_work = True

            if not found_work:
                print("ERROR: Circular dependency or unresolvable graph detected!")
                break

        return results

def main() -> None:
    print("--- Lesson 10: Dependency Graphs ---")
    try:
        bot = DependencyGraphAgent()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)

    while True:
        try:
            user_request = input("\nWhat complex project should I map out? (e.g., 'Add 2 and 3, then add 4 and 5, then multiply the results'): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_request:
            continue

        if user_request.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            print("\nAI is generating the dependency graph...")
            tasks = bot.generate_graph(user_request)
            
            print("\n--- DEPENDENCY GRAPH ---")
            for t in tasks:
                deps = t.get("depends_on", [])
                print(f"Task {t['id']}: {t['description']} (Depends on: {deps if deps else 'None'})")

            print("\n--- EXECUTING GRAPH ---")
            results = bot.execute_graph(tasks)

            print("\n--- FINAL OUTPUT ---")
            for tid, res in results.items():
                print(f"Task {tid} Final Value: {res}")
        except Exception as e:
            print(f"Execution Error: {e}")

if __name__ == "__main__":
    main()

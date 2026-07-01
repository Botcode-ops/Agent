# pyrefly: ignore [missing-import]
from llama_cpp import Llama, LlamaGrammar
from typing import List, Dict, Any
import json
import sys

class ReliabilityAgent:
    """An agent evaluator framework utilizing LLM-as-a-Judge for auto-correction/reliability loops."""

    def __init__(self, model_path: str = "myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf") -> None:
        self.llm = Llama(model_path=model_path)
        grammar_text = r"""
        root   ::= object
        object ::= "{" ws "\"passed\":" ws boolean "," ws "\"score\":" ws number "," ws "\"feedback\":" ws string ws "}"
        boolean ::= "true" | "false"
        string ::= "\"" ([^"\\\\] | "\\" ["\\/bfnrt] | "\\" "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])* "\""
        number ::= [0-9] | "10"
        ws     ::= [ \t\n\r]*
        """
        self.eval_grammar = LlamaGrammar.from_string(grammar_text)

    def run_worker(self, prompt: str) -> str:
        """Execute worker agent response to solve a task."""
        print(f"\n[WORKER] Solving: {prompt}")
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}]
        )
        return str(response["choices"][0]["message"]["content"])

    def run_evaluator(self, task: str, answer: str) -> Dict[str, Any]:
        """Grade the worker output and return detailed feedback."""
        print(f"\n[EVALUATOR] Grading task: {task}")
        eval_prompt = (
            f"TASK: {task}\n"
            f"AGENT_ANSWER: {answer}\n\n"
            "Your job is to evaluate if the agent answered correctly. \n"
            "Be strict. Provide a score from 0 to 10."
        )
        
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": eval_prompt}],
            grammar=self.eval_grammar
        )
        return dict(json.loads(response["choices"][0]["message"]["content"]))

def main() -> None:
    print("--- Lesson 11: Testing & Reliability (LLM-as-a-Judge) ---")
    try:
        bot = ReliabilityAgent()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)

    while True:
        try:
            user_task = input("\nProvide a task for the AI to solve and be tested on: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_task:
            continue

        if user_task.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # --- STEP 1: WORKER PHASE ---
        worker_answer = bot.run_worker(user_task)
        print(f"Worker Output: {worker_answer}")

        # --- STEP 2: EVALUATION PHASE ---
        print("\nAI is evaluating the output...")
        evaluation = bot.run_evaluator(user_task, worker_answer)

        print("\n--- EVALUATION RESULTS ---")
        print(f"PASSED: {evaluation.get('passed')}")
        print(f"SCORE: {evaluation.get('score')}/10")
        print(f"FEEDBACK: {evaluation.get('feedback')}")

        if not evaluation.get('passed'):
            print("\n[RETRY] The evaluator failed the answer. Attempting a fix...")
            fix_prompt = f"Your previous answer was: {worker_answer}. Feedback: {evaluation.get('feedback')}. Please provide a better answer."
            fixed_answer = bot.run_worker(fix_prompt)
            print(f"Fixed Output: {fixed_answer}")
            
            print("\nRe-evaluating fix...")
            second_eval = bot.run_evaluator(user_task, fixed_answer)
            print(f"NEW SCORE: {second_eval.get('score')}/10")

if __name__ == "__main__":
    main()

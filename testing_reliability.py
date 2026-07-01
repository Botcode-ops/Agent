# pyrefly: ignore [missing-import]
from llama_cpp import Llama, LlamaGrammar
import json

# 1. Setup the Model
llm = Llama(model_path="myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# 2. Grammar for Evaluation Output
# We want: {"passed": true/false, "score": 0-10, "feedback": "..."}
GRAMMAR_TEXT = r"""
root   ::= object
object ::= "{" ws "\"passed\":" ws boolean "," ws "\"score\":" ws number "," ws "\"feedback\":" ws string ws "}"
boolean ::= "true" | "false"
string ::= "\"" ([^"\\\\] | "\\" ["\\/bfnrt] | "\\" "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])* "\""
number ::= [0-9] | "10"
ws     ::= [ \t\n\r]*
"""
eval_grammar = LlamaGrammar.from_string(GRAMMAR_TEXT)

def run_worker(prompt):
    print(f"\n[WORKER] Solving: {prompt}")
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

def run_evaluator(task, answer):
    print(f"\n[EVALUATOR] Grading task: {task}")
    eval_prompt = f"""
    TASK: {task}
    AGENT_ANSWER: {answer}
    
    Your job is to evaluate if the agent answered correctly. 
    Be strict. Provide a score from 0 to 10.
    """
    
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": eval_prompt}],
        grammar=eval_grammar
    )
    return json.loads(response["choices"][0]["message"]["content"])

print("--- Lesson 11: Testing & Reliability (LLM-as-a-Judge) ---")

while True:
    user_task = input("\nProvide a task for the AI to solve and be tested on: ")
    if not user_task: break

    # --- STEP 1: WORKER PHASE ---
    worker_answer = run_worker(user_task)
    print(f"Worker Output: {worker_answer}")

    # --- STEP 2: EVALUATION PHASE ---
    print("\nAI is evaluating the output...")
    evaluation = run_evaluator(user_task, worker_answer)

    print("\n--- EVALUATION RESULTS ---")
    print(f"PASSED: {evaluation['passed']}")
    print(f"SCORE: {evaluation['score']}/10")
    print(f"FEEDBACK: {evaluation['feedback']}")

    if not evaluation['passed']:
        print("\n[RETRY] The evaluator failed the answer. Attempting a fix...")
        # In a real system, we'd feed the feedback back to the worker
        fix_prompt = f"Your previous answer was: {worker_answer}. Feedback: {evaluation['feedback']}. Please provide a better answer."
        fixed_answer = run_worker(fix_prompt)
        print(f"Fixed Output: {fixed_answer}")
        
        print("\nRe-evaluating fix...")
        second_eval = run_evaluator(user_task, fixed_answer)
        print(f"NEW SCORE: {second_eval['score']}/10")

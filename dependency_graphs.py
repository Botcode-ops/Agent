from llama_cpp import Llama, LlamaGrammar
import json

# 1. Setup the Model
llm = Llama(model_path="myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# 2. Grammar for a Dependency Graph
# We want: {"tasks": [{"id": 1, "task": "...", "depends_on": []}, ...]}
GRAMMAR_TEXT = r"""
root   ::= object
object ::= "{" ws "\"tasks\":" ws "[" ws task_list ws "]" ws "}"
task_list ::= task (ws "," ws task)*
task   ::= "{" ws "\"id\":" ws number "," ws "\"description\":" ws string "," ws "\"depends_on\":" ws "[" ws id_list ws "]" ws "}"
id_list ::= (number (ws "," ws number)*)?
string ::= "\"" ([^"\\\\] | "\\" ["\\/bfnrt] | "\\" "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])* "\""
number ::= [0-9]+
ws     ::= [ \t\n\r]*
"""
graph_grammar = LlamaGrammar.from_string(GRAMMAR_TEXT)

# 3. Simple Toolset
def calculate(operation, a, b):
    print(f"\n[SYSTEM: Executing {operation} on {a} and {b}]")
    if operation == "add": return a + b
    if operation == "multiply": return a * b
    return 0

print("--- Lesson 10: Dependency Graphs ---")

while True:
    user_request = input("\nWhat complex project should I map out? (e.g., 'Add 2 and 3, then add 4 and 5, then multiply the results'): ")
    if not user_request: break

    # --- STEP 1: GENERATE THE GRAPH ---
    print("\nAI is generating the dependency graph...")
    prompt = f"""Break this request into a dependency graph: {user_request}
    Tasks should be atomic math operations. Use the 'id' to reference dependencies."""
    
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        grammar=graph_grammar
    )
    
    graph_data = json.loads(response["choices"][0]["message"]["content"])
    tasks = graph_data["tasks"]
    
    print("\n--- DEPENDENCY GRAPH ---")
    for t in tasks:
        deps = t['depends_on']
        print(f"Task {t['id']}: {t['description']} (Depends on: {deps if deps else 'None'})")

    # --- STEP 2: EXECUTE THE GRAPH ---
    print("\n--- EXECUTING GRAPH ---")
    results = {} # Store results by task ID
    
    # Simple topological sort/execution loop
    completed_ids = set()
    while len(completed_ids) < len(tasks):
        found_work = False
        for t in tasks:
            if t['id'] in completed_ids:
                continue
            
            # Check if all dependencies are met
            if all(dep_id in completed_ids for dep_id in t['depends_on']):
                print(f"\nRunning Task {t['id']}: {t['description']}")
                
                # Use Worker to solve the specific task using previous results
                context = "\n".join([f"Result of Task {rid}: {res}" for rid, res in results.items() if rid in t['depends_on']])
                
                worker_prompt = f"""Solve this task: {t['description']}
                Previous Context: {context}
                Provide ONLY the numeric result."""
                
                worker_response = llm.create_chat_completion(
                    messages=[{"role": "user", "content": worker_prompt}]
                )
                
                res_text = worker_response["choices"][0]["message"]["content"].strip()
                # Try to extract just the number if AI was chatty
                try:
                    res_val = float(''.join(c for c in res_text if c.isdigit() or c == '.'))
                    results[t['id']] = res_val
                    print(f"Result: {res_val}")
                except:
                    results[t['id']] = res_text
                    print(f"Result: {res_text}")
                    
                completed_ids.add(t['id'])
                found_work = True
        
        if not found_work:
            print("ERROR: Circular dependency or unresolvable graph!")
            break

    print("\n--- FINAL OUTPUT ---")
    for tid, res in results.items():
        print(f"Task {tid} Final Value: {res}")

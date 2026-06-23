from llama_cpp import Llama, LlamaGrammar
import json

# 1. Setup the Model
llm = Llama(model_path="myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# 2. Define the Grammar (Force JSON with specific keys)
# The agent must output: {"thought": "...", "action": "...", "params": {...}}
GRAMMAR_TEXT = r"""
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
structured_grammar = LlamaGrammar.from_string(GRAMMAR_TEXT)

# 3. Tools (The actions the agent can take)
def calculate(a, b, operation):
    print(f"\n[SYSTEM: Executing {operation} on {a} and {b}]")
    try:
        a, b = float(a), float(b)
        if operation == "add": return str(a + b)
        if operation == "multiply": return str(a * b)
    except:
        return "Error: Invalid numbers"
    return "0"

print("--- Lesson 9: Structured Actions (JSON Agent) ---")
print("Available Actions: 'calculate' (params: a, b, operation), 'final_answer' (params: answer)")

while True:
    user_request = input("\nUser: ")
    if not user_request: break

    history = [
        {
            "role": "system", 
            "content": (
                "You are a structured agent. You MUST respond with JSON containing 'thought', 'action', and 'params'.\n"
                "Available actions:\n"
                "- calculate: a, b, operation ('add', 'multiply')\n"
                "- final_answer: answer\n"
                "Example: {\"thought\": \"I need to add these\", \"action\": \"calculate\", \"params\": {\"a\": 5, \"b\": 3, \"operation\": \"add\"}}"
            )
        },
        {"role": "user", "content": user_request}
    ]

    # Autonomous loop
    for step in range(5):
        print(f"\n--- [Step {step + 1}] ---")
        
        response = llm.create_chat_completion(
            messages=history,
            grammar=structured_grammar
        )
        
        output_text = response["choices"][0]["message"]["content"]
        
        try:
            action_obj = json.loads(output_text)
            thought = action_obj.get("thought")
            action = action_obj.get("action")
            params = action_obj.get("params", {})

            print(f"THOUGHT: {thought}")
            print(f"ACTION: {action}({params})")

            if action == "final_answer":
                print(f"AI FINAL ANSWER: {params.get('answer')}")
                break
            
            if action == "calculate":
                result = calculate(params.get("a"), params.get("b"), params.get("operation"))
                print(f"OBSERVATION: {result}")
                
                # Add the thought, action, and result back to history
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

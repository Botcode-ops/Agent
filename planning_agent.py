from llama_cpp import Llama, LlamaGrammar
import json

# 1. Setup the Model
llm = Llama(model_path="myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# 2. The Planner Prompt & Grammar
PLANNER_SYSTEM_PROMPT = """
You are a planning assistant. Your job is to take a user request and break it 
down into exactly 3 logical steps. 

Format your response ONLY as a JSON list of strings.
Example: ["Step 1", "Step 2", "Step 3"]
"""

# Force exactly a list of 3 strings
GRAMMAR_TEXT = r"""
root   ::= "[" ws string "," ws string "," ws string ws "]"
string ::= "\"" ([^"\\\\] | "\\" ["\\/bfnrt] | "\\" "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])* "\""
ws     ::= [ \t\n\r]*
"""
planner_grammar = LlamaGrammar.from_string(GRAMMAR_TEXT)

# 3. The Tool (A simple math tool for the AI to "plan" around)
def calculate(operation, a, b):
    print(f"\n[SYSTEM: Executing {operation} on {a} and {b}]")
    if operation == "add": return str(a + b)
    if operation == "multiply": return str(a * b)
    return "0"

tools = [
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

print("--- Lesson 8: The Planning Agent ---")

while True:
    user_request = input("\nWhat complex task should I plan? (e.g., 'Add 5 and 10 then multiply by 2'): ")
    if not user_request: break

    # --- STEP 1: GENERATE THE PLAN ---
    print("\nAI is creating a plan...")
    plan_messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Create a plan for: {user_request}"}
    ]
    
    response = llm.create_chat_completion(messages=plan_messages, grammar=planner_grammar)
    plan_text = response["choices"][0]["message"]["content"]

    try:
        plan = json.loads(plan_text)
        print(f"--- THE PLAN ---")
        for i, step in enumerate(plan):
            print(f"{i+1}. {step}")
        
        # --- STEP 2: EXECUTE THE PLAN ---
        print("\n--- EXECUTING PLAN ---")
        context = ""
        
        for step in plan:
            print(f"\nCurrent Task: {step}")
            
            worker_messages = [
                {"role": "system", "content": f"You are a worker. Solve this step: {step}. Context: {context}"},
                {"role": "user", "content": "Use the calculate tool if needed, otherwise provide the answer."}
            ]
            
            # The worker can use tools
            step_response = llm.create_chat_completion(messages=worker_messages, tools=tools)
            message = step_response["choices"][0]["message"]

            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    fn_name = tool_call["function"]["name"]
                    args = json.loads(tool_call["function"]["arguments"])
                    
                    if fn_name == "calculate":
                        result = calculate(args["a"], args["b"], args["operation"])
                        print(f"Tool Result: {result}")
                        
                        # Add tool result to context
                        context += f"\nResult of '{step}' (via tool): {result}"
            else:
                step_result = message["content"]
                print(f"Task Result: {step_result}")
                context += f"\nResult of '{step}': {step_result}"

        # --- STEP 3: FINAL SUMMARY ---
        print("\n--- FINAL ANSWER ---")
        final_messages = [
            {"role": "system", "content": "Summarize the final result based on these completed steps."},
            {"role": "user", "content": context}
        ]
        summary_response = llm.create_chat_completion(messages=final_messages)
        print(summary_response["choices"][0]["message"]["content"])

    except Exception as e:
        print(f"Error: {e}")
        print(f"AI said: {plan_text}")


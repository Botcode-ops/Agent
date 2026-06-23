from llama_cpp import Llama
import json

# 1. Setup the Model
llm = Llama(model_path="myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# 2. Define the Tools
def calculate(a, b, operation):
    print(f"\n[System: Executing calculate {a} {operation} {b}]")
    if operation == "add": return str(a + b)
    if operation == "multiple": return str(a * b)
    return "Error"

tools = [
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

# 3. The ReAct System Prompt
system_prompt = """
You are an autonomous agent. Solve the user's request step-by-step.
For each step, you must follow this format:

THOUGHT: (Explain what you are doing)
ACTION: (Call a tool if needed)

When you have the final result, you MUST say:
FINAL ANSWER: (Your final response)
"""

# 4. The Main Loop
while True:
    user_text = input("\nUser Request: ")
    if not user_text: continue
    
    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text}
    ]

    # --- THE AUTONOMOUS LOOP (Max 5 steps) ---
    for i in range(5):
        print(f"--- [AI Thinking Step {i+1}] ---")
        
        response = llm.create_chat_completion(messages=history, tools=tools)
        message = response["choices"][0]["message"]
        
        # Check if the AI is finished
        if message.get("content") and "FINAL ANSWER" in message["content"]:
            print(f"AI: {message['content']}")
            break
        
        # Handle tool calls (The "ACTION" and "OBSERVATION")
        if message.get("tool_calls"):
            history.append(message)
            for tool_call in message["tool_calls"]:
                fn_name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"])
                
                result = calculate(args["a"], args["b"], args["operation"])
                
                history.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": fn_name,
                    "content": result
                })
                print(f"Observation: Result is {result}")
        else:
            # If no tool call and no final answer, just add the thought to history
            history.append(message)
            if message.get("content"):
                print(f"AI Thought: {message['content']}")

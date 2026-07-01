from llama_cpp import Llama
import json

# 1. Setup the Model 
llm = Llama(model_path="myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# 2. Define the Python Functions (The "Abilities")
def get_weather(city):
    print(f"\n[System: Calling get_weather for {city}]")
    return "It is rainy."

def calculate(a, b, operation):
    print(f"\n[System: Calling calculate for {operation}]")
    if operation == "add":
        return str(a + b)
    elif operation == "multiple":
        return str(a * b)
    return "Error: Unknown operation"

# 3. The Tool Blueprints (The "Manual")
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform math like add or multiple",
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

# 4. Initialize History
history = [{"role": "system", "content": "You are a helpful assistant with tools."}]

# 5. The Chat Loop
while True:
    user_text = input("\nYou: ")
    if not user_text: continue
    history.append({"role": "user", "content": user_text})

    response = llm.create_chat_completion(messages=history, tools=tools)
    message = response["choices"][0]["message"]

    if message.get("tool_calls"):
        history.append(message)
        
        for tool_call in message["tool_calls"]:
            fn_name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])
            
            if fn_name == "get_weather":
                result = get_weather(args["city"])
            elif fn_name == "calculate":
                result = calculate(args["a"], args["b"], args["operation"])
            
            history.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": fn_name,
                "content": result
            })

        # Get final response from AI
        final_response = llm.create_chat_completion(messages=history)
        ai_text = final_response["choices"][0]["message"]["content"]
        print(f"AI: {ai_text}")
        history.append({"role": "assistant", "content": ai_text})
    else:
        ai_text = message["content"]
        if ai_text:
            print(f"AI: {ai_text}")
            history.append({"role": "assistant", "content": ai_text})
        else:
            print("AI: (No response)")

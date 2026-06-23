from llama_cpp import Llama
import json
import os

# 1. Setup the Model
llm = Llama(model_path="myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

MEMORY_FILE = "memory.json"

# 2. Check for Long-term Memory
if os.path.exists(MEMORY_FILE):
    # Load the memory
    with open(MEMORY_FILE, "r") as file:
        memory_data = json.load(file)
    user_name = memory_data.get("user_name", "Friend")
    print(f"--- [SYSTEM: Memory Loaded! Welcome back, {user_name}] ---")
else:
    # Create new memory
    print("AI: Hello! I don't believe we've met. What is your name?")
    user_name = input("You (Your Name): ")
    
    memory_data = {"user_name": user_name}
    with open(MEMORY_FILE, "w") as file:
        json.dump(memory_data, file)
    print(f"--- [SYSTEM: Memory Saved! Nice to meet you, {user_name}] ---")

# 3. Setup the Chat with Memory
history = [
    {
        "role": "system", 
        "content": f"You are a helpful assistant. The user's name is {user_name}. Always be polite to them."
    }
]

# 4. The Chat Loop
while True:
    user_text = input(f"\n{user_name}: ")
    if not user_text: continue
    history.append({"role": "user", "content": user_text})

    response = llm.create_chat_completion(messages=history)
    ai_text = response["choices"][0]["message"]["content"]
    
    print(f"AI: {ai_text}")
    history.append({"role": "assistant", "content": ai_text})

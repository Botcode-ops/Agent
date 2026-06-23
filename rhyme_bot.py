from llama_cpp import Llama

# 1. Initialize the model
llm = Llama(model_path="myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
history = []

# 2. Add the SYSTEM PROMPT (The "Rules")
# This tells the AI how to behave throughout the whole conversation.
history.append({"role": "system", "content": "You are a helpful assistant that only answers in rhymes."})

# 3. The Chat Loop
while True:
    # Get user message
    user_text = input("You: ")
    history.append({"role": "user", "content": user_text})
    
    # Get AI response
    response = llm.create_chat_completion(messages=history)
    ai_text = response["choices"][0]["message"]["content"]
    
    # Add AI response to history and print it
    history.append({"role": "assistant", "content": ai_text})
    print(ai_text)

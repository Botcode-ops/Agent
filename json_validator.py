from llama_cpp import Llama
import json

# 1. Setup the Model
llm = Llama(model_path="myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
history = []

# 2. Set the System Prompt
# This "primes" the model to only output data, not conversational text.
history.append({
    "role": "system", 
    "content": "You are a data assistant. You ONLY output valid JSON. Do not include any other text."
})

print("Assistant: I am ready to give you structured JSON data!")

while True:
    # 3. Get User Input
    user_text = input("You: ")
    history.append({"role": "user", "content": user_text})
    
    # 4. The Retry Loop (3 Attempts)
    for i in range(3):
        print(f"--- Attempt {i+1} ---")
        
        # Get response from AI
        response = llm.create_chat_completion(messages=history)
        ai_text = response["choices"][0]["message"]["content"]
        
        # 5. The Validation Step
        try:
            # Check if it's valid JSON
            data = json.loads(ai_text)
            
            print("Success! Valid JSON found:")
            print(data)
            
            # Save valid response to history and EXIT the for-loop
            history.append({"role": "assistant", "content": ai_text})
            break 
            
        except:
            # If it failed, this runs instead of the 'break'
            print("Error: The AI didn't send valid JSON. Let's try again...")
            
            # 6. Check if this was the last attempt
            if i == 2:
                print("Final attempt failed. Please try a different question.")

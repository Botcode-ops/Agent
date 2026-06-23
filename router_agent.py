from llama_cpp import Llama
import json

# 1. Setup the Model
llm = Llama(model_path="myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# 2. The Receptionist (Classification Prompt)
classifier_prompt = """
You are a routing assistant. Your job is to read the user's message 
and pick the correct category. Only output the category name.

CATEGORIES:
- MATH: For calculations and numbers.
- WEATHER: For questions about rain, sun, or temperature.
- GENERAL: For greetings, chat, or anything else.

Response format: ONLY the category name in uppercase.
"""

# 3. The Specialist Prompts
math_expert_prompt = "You are a math expert. Provide clear, step-by-step solutions to math problems."

print("Assistant: I am ready! I will route your request to the right specialist.")

while True:
    user_text = input("\nYou: ")
    if not user_text: continue
    
    # --- STEP 1: CLASSIFICATION ---
    classification_messages = [
        {"role": "system", "content": classifier_prompt},
        {"role": "user", "content": user_text}
    ]
    
    response = llm.create_chat_completion(messages=classification_messages)
    category = response["choices"][0]["message"]["content"].strip().upper()
    
    print(f"--- [RECEPTIONIST: Routing to {category}] ---")
    
    # --- STEP 2: ROUTING & SPECIALISTS ---
    if "MATH" in category:
        # Route to Math Specialist
        messages = [
            {"role": "system", "content": math_expert_prompt},
            {"role": "user", "content": user_text}
        ]
        final_response = llm.create_chat_completion(messages=messages)
        print(f"AI (Math Expert): {final_response['choices'][0]['message']['content']}")
        
    elif "WEATHER" in category:
        # Route to Weather Specialist (Direct Code)
        print("AI (Weather System): It is currently rainy!")
        
    else:
        # Route to General Specialist (Your Greeting!)
        print("AI: Hello! How can I help you today?")

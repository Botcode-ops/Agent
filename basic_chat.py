from llama_cpp import Llama
llm = Llama(model_path = "myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
response = llm("the moon made of",max_tokens=25)
print(response["choices"][0]["text"])
history = []
while True:
    user_text = input("You: ")
    history.append({"role": "user", "content": user_text})
    response = llm.create_chat_completion(messages=history)
    ai_text = response["choices"][0]["message"]["content"]
    history.append({"role": "assistant", "content": ai_text})
    print(ai_text)


from llama_cpp import Llama
import json
import time
import uuid

# 1. Setup the Model
llm = Llama(model_path="myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# 2. The Observability Logger
class AgentTracer:
    def __init__(self):
        self.trace_id = str(uuid.uuid4())
        self.logs = []
        print(f"--- [OBSERVABILITY: New Trace Started {self.trace_id}] ---")

    def log_event(self, role, content, metadata=None):
        event = {
            "timestamp": time.time(),
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        self.logs.append(event)
        print(f"[{role.upper()}] {content[:100]}...")

    def save_trace(self):
        filename = f"trace_{self.trace_id}.json"
        with open(filename, "w") as f:
            json.dump(self.logs, f, indent=2)
        print(f"\n--- [OBSERVABILITY: Trace Saved to {filename}] ---")

# 3. Instrumented LLM Call
def tracked_completion(tracer, messages):
    start_time = time.time()
    
    response = llm.create_chat_completion(messages=messages)
    
    end_time = time.time()
    latency = end_time - start_time
    
    ai_message = response["choices"][0]["message"]["content"]
    
    # Extract token usage if available
    usage = response.get("usage", {})
    
    tracer.log_event("assistant", ai_message, {
        "latency_sec": round(latency, 2),
        "tokens": usage.get("total_tokens", 0)
    })
    
    return ai_message

print("--- Lesson 12: Observability (Tracing & Monitoring) ---")

while True:
    user_input = input("\nAsk the agent something: ")
    if not user_input: break

    # Initialize a new trace for this interaction
    tracer = AgentTracer()
    tracer.log_event("user", user_input)

    history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input}
    ]

    # Run the completion with tracking
    answer = tracked_completion(tracer, history)
    
    print(f"\nAI: {answer}")

    # Finalize the trace
    tracer.save_trace()

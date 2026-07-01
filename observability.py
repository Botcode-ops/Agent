# pyrefly: ignore [missing-import]
from llama_cpp import Llama
from typing import List, Dict, Any, Optional
import json
import time
import uuid
import sys

class AgentTracer:
    """Logs and traces events, latency, and token consumption of agent operations."""

    def __init__(self) -> None:
        self.trace_id: str = str(uuid.uuid4())
        self.logs: List[Dict[str, Any]] = []
        print(f"--- [OBSERVABILITY: New Trace Started {self.trace_id}] ---")

    def log_event(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a single execution event within the trace."""
        event = {
            "timestamp": time.time(),
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        self.logs.append(event)
        print(f"[{role.upper()}] {content[:100]}...")

    def save_trace(self, folder_path: str = ".") -> None:
        """Persist trace logs to a local JSON file."""
        filename = f"{folder_path}/trace_{self.trace_id}.json"
        try:
            with open(filename, "w") as f:
                json.dump(self.logs, f, indent=2)
            print(f"\n--- [OBSERVABILITY: Trace Saved to {filename}] ---")
        except Exception as e:
            print(f"--- [OBSERVABILITY: Error saving trace: {e}] ---")

class InstrumentedAgent:
    """An agent that interacts with the Llama engine with automatic event tracing and performance logging."""

    def __init__(self, model_path: str = "myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf") -> None:
        self.llm = Llama(model_path=model_path)

    def tracked_completion(self, tracer: AgentTracer, messages: List[Dict[str, str]]) -> str:
        """Execute chat completion with timed latency tracking and token consumption checks."""
        start_time = time.time()
        response = self.llm.create_chat_completion(messages=messages)
        end_time = time.time()
        
        latency = end_time - start_time
        ai_message = str(response["choices"][0]["message"]["content"])
        usage = response.get("usage", {})
        
        tracer.log_event("assistant", ai_message, {
            "latency_sec": round(latency, 2),
            "tokens": usage.get("total_tokens", 0)
        })
        
        return ai_message

def main() -> None:
    print("--- Lesson 12: Observability (Tracing & Monitoring) ---")
    try:
        agent = InstrumentedAgent()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)

    while True:
        try:
            user_input = input("\nAsk the agent something: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Initialize a new trace for this interaction
        tracer = AgentTracer()
        tracer.log_event("user", user_input)

        history = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ]

        # Run the completion with tracking
        answer = agent.tracked_completion(tracer, history)
        print(f"\nAI: {answer}")

        # Finalize the trace
        tracer.save_trace()

if __name__ == "__main__":
    main()

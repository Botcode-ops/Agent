# pyrefly: ignore [missing-import]
"""
Master Agent System.

An agent orchestration framework utilizing llama-cpp-python and GBNF grammars
to dynamically route query processing to specialized Math and Information agents.
"""

from llama_cpp import Llama, LlamaGrammar
import json
import sys
from typing import Dict, Any, Tuple, Optional

# --- CONFIGURATION ---
MODEL_PATH = "myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# --- GBNF GRAMMAR DEFINITIONS ---

ROUTER_GRAMMAR_SRC = r"""
root ::= rtmain
rtmain ::= "{" ws "\"thought\"" ws ":" ws rtstr "," ws "\"agent\"" ws ":" ws rtagent "}"
rtagent ::= "\"math\"" | "\"info\"" | "\"none\""
rtstr ::= "\"" ([^"\\\\] | "\\" ["\\/bfnrt] | "\\" "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])* "\""
ws ::= [ \t\n\r]*
"""

MATH_GRAMMAR_SRC = r"""
root ::= mtmain
mtmain ::= "{" ws "\"thought\"" ws ":" ws mtstr "," ws "\"action\"" ws ":" ws mtact "," ws "\"params\"" ws ":" ws mtprm ws "}"
mtact ::= "\"calculate\"" | "\"final_answer\""
mtprm ::= mtobj | "null"
mtobj ::= "{" ws (mtstr ws ":" ws mtval (ws "," ws mtstr ws ":" ws mtval)*)? ws "}"
mtval ::= mtobj | mtarr | mtstr | mtnum | "true" | "false" | "null"
mtarr ::= "[" ws (mtval (ws "," ws mtval)*)? ws "]"
mtstr ::= "\"" ([^"\\\\] | "\\" ["\\/bfnrt] | "\\" "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])* "\""
mtnum ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws ::= [ \t\n\r]*
"""

INFO_GRAMMAR_SRC = r"""
root ::= itmain
itmain ::= "{" ws "\"thought\"" ws ":" ws itstr "," ws "\"response\"" ws ":" ws itstr "}"
itstr ::= "\"" ([^"\\\\] | "\\" ["\\/bfnrt] | "\\" "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])* "\""
ws ::= [ \t\n\r]*
"""


# --- TOOLS ---

def calculate(a: Any, b: Any, operation: str) -> str:
    """
    Perform mathematical addition or multiplication.
    
    Args:
        a: First numeric value (float-convertible).
        b: Second numeric value (float-convertible).
        operation: Calculation operation ('add' or 'multiply').
        
    Returns:
        String result of the calculation or error message.
    """
    print(f"\n[MATH TOOL: {operation} {a} and {b}]")
    try:
        num_a = float(a)
        num_b = float(b)
        op = str(operation).lower().strip()
        if op == "add":
            return str(num_a + num_b)
        elif op == "multiply":
            return str(num_a * num_b)
        return f"Unsupported operation: {op}"
    except (ValueError, TypeError) as e:
        return f"Error: {e}"


# --- AGENTS ---

class MathAgent:
    """Specialist agent designed to resolve mathematical problems using external tools."""

    def __init__(self, llm: Llama, grammar: LlamaGrammar) -> None:
        self.llm = llm
        self.grammar = grammar

    def run(self, user_query: str) -> str:
        """
        Execute the math agent reasoning and action loop.
        
        Args:
            user_query: The math-related problem query.
            
        Returns:
            The calculated final answer or a timeout error.
        """
        print("\n[HANDOVER: Math Agent taking control...]")
        history = [
            {
                "role": "system",
                "content": (
                    "You are a Math Specialist. Use the 'calculate' tool.\n"
                    "calculate params: {'a': number, 'b': number, 'operation': 'add' or 'multiply'}\n"
                    "Example: {'thought': 'Adding', 'action': 'calculate', 'params': {'a': 1, 'b': 2, 'operation': 'add'}}\n"
                    "When done, use 'final_answer'."
                )
            },
            {"role": "user", "content": user_query}
        ]

        for _ in range(3):
            try:
                res = self.llm.create_chat_completion(
                    messages=history,
                    grammar=self.grammar,
                    max_tokens=256
                )
                output = json.loads(res["choices"][0]["message"]["content"])
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                return f"Math agent error during completion decoding: {e}"

            print(f"MATH THOUGHT: {output.get('thought')}")

            action = output.get("action")
            params = output.get("params", {})

            if action == "final_answer":
                return str(params.get("answer", "No answer provided in parameters"))

            if action == "calculate":
                obs = calculate(params.get("a"), params.get("b"), params.get("operation"))
                print(f"OBSERVATION: {obs}")
                history.append({"role": "assistant", "content": json.dumps(output)})
                history.append({"role": "user", "content": f"Observation: {obs}"})
            else:
                return f"Math agent error: Invalid action '{action}' selected."

        return "Math agent timed out."


class InfoAgent:
    """Specialist agent designed to query and retrieve general knowledge facts."""

    def __init__(self, llm: Llama, grammar: LlamaGrammar) -> None:
        self.llm = llm
        self.grammar = grammar

    def run(self, user_query: str) -> str:
        """
        Query the information specialist agent for a single-sentence fact.
        
        Args:
            user_query: The factual user query.
            
        Returns:
            A single sentence response containing the fact.
        """
        print("\n[HANDOVER: Information Agent taking control...]")
        try:
            res = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are an Information Specialist. Provide a ONE SENTENCE fact."},
                    {"role": "user", "content": user_query}
                ],
                grammar=self.grammar,
                max_tokens=400
            )
            output = json.loads(res["choices"][0]["message"]["content"])
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            return f"Information agent decoding error: {e}"

        print(f"INFO THOUGHT: {output.get('thought')}")
        return str(output.get("response", "No response found"))


class RouterAgent:
    """Receptionist agent that determines which agent should process the user query."""

    def __init__(self, llm: Llama, grammar: LlamaGrammar) -> None:
        self.llm = llm
        self.grammar = grammar

    def route(self, user_query: str) -> Tuple[str, str]:
        """
        Determine domain routing ('math', 'info', 'none') for the user query.
        
        Args:
            user_query: Input request text.
            
        Returns:
            A tuple of (routed_agent_name, agent_thought).
        """
        res = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Router Agent. Categorize the user request.\n"
                        "Rules:\n"
                        "- agent='math' if the query has numbers and needs calculation (e.g., 'plus', 'multiplied', 'root').\n"
                        "- agent='info' if the query is a question about facts or general topics.\n"
                        "Examples:\n"
                        "User: '2 + 2'\n"
                        "JSON: {\"thought\": \"Mathematical addition\", \"agent\": \"math\"}\n"
                        "User: 'Who is Einstein?'\n"
                        "JSON: {\"thought\": \"Biography request\", \"agent\": \"info\"}\n"
                    )
                },
                {"role": "user", "content": user_query}
            ],
            grammar=self.grammar,
            max_tokens=100
        )
        output = json.loads(res["choices"][0]["message"]["content"])
        return str(output.get("agent", "none")), str(output.get("thought", ""))


# --- MASTER ORCHESTRATOR ---

class MasterAgentSystem:
    """Primary system manager coordinating routing and execution of sub-agents."""

    def __init__(self, model_path: str = MODEL_PATH) -> None:
        print(f"Initializing Llama engine with model: {model_path}")
        self.llm = Llama(model_path=model_path, n_ctx=2048)
        
        # Load Grammars
        self.g_router = LlamaGrammar.from_string(ROUTER_GRAMMAR_SRC)
        self.g_math = LlamaGrammar.from_string(MATH_GRAMMAR_SRC)
        self.g_info = LlamaGrammar.from_string(INFO_GRAMMAR_SRC)
        
        # Instantiate Sub-agents
        self.router_agent = RouterAgent(self.llm, self.g_router)
        self.math_agent = MathAgent(self.llm, self.g_math)
        self.info_agent = InfoAgent(self.llm, self.g_info)

    def handle_query(self, user_query: str) -> str:
        """
        Route and execute the query, returning the final computed response.
        
        Args:
            user_query: The incoming user question.
            
        Returns:
            The final answer or error details.
        """
        print("\n[MASTER: Routing query...]")
        try:
            agent, thought = self.router_agent.route(user_query)
        except Exception as e:
            return f"ROUTER ERROR: Failed to classify query. {e}"

        print(f"ROUTER THOUGHT: {thought}")
        print(f"DECISION: Route to {agent}")

        try:
            if agent == "math":
                return self.math_agent.run(user_query)
            elif agent == "info":
                return self.info_agent.run(user_query)
            else:
                return "I'm not sure how to help with that."
        except Exception as e:
            return f"Agent Error: {e}"


def main() -> None:
    """Interactive loop entrypoint."""
    print("--- Master Agent System ---")
    print("Type 'exit' or 'quit' to end the session.")
    
    try:
        system = MasterAgentSystem()
    except Exception as e:
        print(f"Initialization Error: Could not load the agent system models. {e}")
        sys.exit(1)

    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting. Goodbye!")
            break

        final_res = system.handle_query(user_input)
        print(f"\nFINAL SYSTEM RESPONSE: {final_res}")


if __name__ == "__main__":
    main()

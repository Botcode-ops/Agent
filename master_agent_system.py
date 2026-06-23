from llama_cpp import Llama, LlamaGrammar
import json

# 1. Setup the Model
# We'll use the same model for all roles but change prompts/grammars
MODEL_PATH = "myenv/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
llm = Llama(model_path=MODEL_PATH, n_ctx=2048)

# --- GRAMMARS ---

# ROUTER_GRAMMAR: Choose which agent to use
ROUTER_GRAMMAR = r"""
root ::= rtmain
rtmain ::= "{" ws "\"thought\"" ws ":" ws rtstr "," ws "\"agent\"" ws ":" ws rtagent "}"
rtagent ::= "\"math\"" | "\"info\"" | "\"none\""
rtstr ::= "\"" ([^"\\\\] | "\\" ["\\/bfnrt] | "\\" "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])* "\""
ws ::= [ \t\n\r]*
"""

# MATH_GRAMMAR (Lesson 9 style)
MATH_GRAMMAR = r"""
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

# INFO_GRAMMAR: Standard structured response
INFO_GRAMMAR = r"""
root ::= itmain
itmain ::= "{" ws "\"thought\"" ws ":" ws itstr "," ws "\"response\"" ws ":" ws itstr "}"
itstr ::= "\"" ([^"\\\\] | "\\" ["\\/bfnrt] | "\\" "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])* "\""
ws ::= [ \t\n\r]*
"""


g_router = LlamaGrammar.from_string(ROUTER_GRAMMAR)
g_math = LlamaGrammar.from_string(MATH_GRAMMAR)
g_info = LlamaGrammar.from_string(INFO_GRAMMAR)

# --- TOOLS ---

def calculate(a, b, operation):
    print(f"\n[MATH TOOL: {operation} {a} and {b}]")
    try:
        a, b = float(a), float(b)
        op = str(operation).lower().strip()
        if op == "add": return str(a + b)
        if op == "multiply": return str(a * b)
        return f"Unsupported operation: {op}"
    except Exception as e:
        return f"Error: {e}"

# --- AGENT LOGIC ---

def run_math_agent(user_query):
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
        res = llm.create_chat_completion(messages=history, grammar=g_math, max_tokens=256)
        output = json.loads(res["choices"][0]["message"]["content"])
        
        print(f"MATH THOUGHT: {output['thought']}")
        
        if output['action'] == "final_answer":
            return output['params'].get('answer')
        
        if output['action'] == "calculate":
            p = output['params']
            obs = calculate(p.get('a'), p.get('b'), p.get('operation'))
            print(f"OBSERVATION: {obs}")
            history.append({"role": "assistant", "content": json.dumps(output)})
            history.append({"role": "user", "content": f"Observation: {obs}"})
    return "Math agent timed out."

def run_info_agent(user_query):
    print("\n[HANDOVER: Information Agent taking control...]")
    res = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are an Information Specialist. Provide a ONE SENTENCE fact."},
            {"role": "user", "content": user_query}
        ],
        grammar=g_info,
        max_tokens=400
    )
    output = json.loads(res["choices"][0]["message"]["content"])
    print(f"INFO THOUGHT: {output['thought']}")
    return output['response']

# --- MASTER AGENT ---

def master_agent():
    print("--- Master Agent System ---")
    print("Type 'exit' to quit.")
    
    while True:
        try:
            user_input = input("\nUser: ")
        except EOFError:
            break
            
        if not user_input or user_input.lower() in ["exit", "quit"]: break
        
        # 1. Routing Phase
        print("\n[MASTER: Routing query...]")
        res = llm.create_chat_completion(
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
                {"role": "user", "content": user_input}
            ],
            grammar=g_router,
            max_tokens=100
        )
        
        try:
            route_obj = json.loads(res["choices"][0]["message"]["content"])
        except Exception as e:
            print(f"ROUTER ERROR: {e}")
            continue

        agent = route_obj.get("agent")
        print(f"ROUTER THOUGHT: {route_obj.get('thought')}")
        print(f"DECISION: Route to {agent}")

        # 2. Execution Phase
        try:
            if agent == "math":
                final_res = run_math_agent(user_input)
            elif agent == "info":
                final_res = run_info_agent(user_input)
            else:
                final_res = "I'm not sure how to help with that."
        except Exception as e:
            final_res = f"Agent Error: {e}"
            
        print(f"\nFINAL SYSTEM RESPONSE: {final_res}")

if __name__ == "__main__":
    master_agent()

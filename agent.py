import os, json, copy
from typing import Any
from pathlib import Path
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

class MessagesState(TypedDict):
    messages: List[BaseMessage]


MODEL_NAME = 'gpt-4.1-nano'
#os.environ['OPENAI_API_KEY'] = 'api key'

llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0)

# Connection check
test_response = llm.invoke([HumanMessage(content='Reply with one word: working?')])
print(f'Model: {MODEL_NAME}')
print(f'Test: {test_response.content}')
print('Setup complete!')

def llm_chat(messages: list, tools: list | None = None) -> AIMessage:
    """
    Sends the message history to the LLM and returns the model response.

    Parameters:
      messages — list of dialog messages. Each message is a LangChain object:
                   SystemMessage(content="...")   — instruction for the model (agent role)
                   HumanMessage(content="...")    — message from the user
                   AIMessage(...)                 — previous model response
                   ToolMessage(content="...", tool_call_id="...") — tool result

      tools   — list of tool descriptions (OpenAI function calling schema or LangChain tools).

    Returns AIMessage:
      msg.content    — text response (str)
      msg.tool_calls — list of tool calls:
                         "name" — tool name
                         "args" — arguments (already parsed dict)
                         "id"   — unique call identifier
    """
    if tools:
        return llm.bind_tools(tools).invoke(messages)
    return llm.invoke(messages)

class ToolTracer:
    """Collects all tool calls."""
    def __init__(self):
        self.calls: list[ToolCallRecord] = []

    def called(self, name: str) -> bool:
        return any(c.name == name for c in self.calls)

    def get_calls(self, name: str) -> list:
        return [c for c in self.calls if c.name == name]

    def print_trace(self) -> None:
        print("=== Tool Call Trace ===")
        for i, c in enumerate(self.calls, 1):
            print(f"  {i}. {c.name}({json.dumps(c.args, ensure_ascii=False)[:80]})")
            if c.result is not None:
                print(f"     -> {json.dumps(c.result, ensure_ascii=False)[:100]}")
        print("=====================")

# ╔══════════════════════════════════════════════════════════════╗
# ║               OUR FRAMEWORK FUNCTIONALITY                    ║
# ╚══════════════════════════════════════════════════════════════╝

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 1. ReAct loop agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def is_on_topic(user_message: str) -> bool:
    """Use LLM to classify whether the message is relevant to coding framework."""
    response = llm.invoke([
        SystemMessage(content=(
            "You are a strict binary classifier.\n"
            "Your task is to decide whether a user message is related to coding.\n\n"

            "Coding includes ONLY:\n"
            "- Writing or understanding source code\n"
            "- Programming concepts (algorithms, data structures)\n"
            "- Software development (backend, frontend, APIs, databases)\n"
            "- Debugging, errors, or technical issues\n"
            "- Development tools, environments, or frameworks\n\n"

            "NOT coding:\n"
            "- General questions\n"
            "- Casual conversation\n"
            "- Non-technical topics\n"
            "- Ambiguous or unclear intent\n\n"

            "Rules:\n"
            "- Respond with ONLY one word: yes OR no\n"
            "- Do NOT explain\n"
            "- Do NOT add punctuation\n"
            "- Do NOT add extra text\n"
            "- If unsure, respond 'no'\n\n"

            "Answer strictly based on the user message."
        )),
        HumanMessage(content=user_message),
    ])
    answer = response.content.strip().lower()
    print(f"[GUARD] is_on_topic → {answer}")
    return answer.startswith("yes")


def input_guard(state: MessagesState) -> MessagesState:
    messages = state["messages"]
    last_msg = messages[-1]
    content = last_msg.content

    has_history = len(messages) > 1
    if has_history and not is_on_topic(content):
        print("[GUARD] 🚫 Blocked")
        return {
            "messages": messages + [
                AIMessage(content="I'm a coding framework. I only help with programming.")
            ]
        }

    return state

def agent_node(state: MessagesState) -> MessagesState:
    messages = state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": messages + [response]
    }

def guard_router(state: MessagesState) -> str:
    last_msg = state["messages"][-1].content

    if "only help with programming" in last_msg:
        return "end"

    return "agent"


builder = StateGraph(MessagesState)

builder.add_node("guard", input_guard)
builder.add_node("agent", agent_node)

builder.set_entry_point("guard")

builder.add_conditional_edges(
    "guard",
    guard_router,
    {
        "agent": "agent",
        "end": END
    }
)

builder.add_edge("agent", END)

graph = builder.compile()

SYSTEM_PROMPT = """
You are an AI coding assistant inside a development framework.
Your goal is to help the user with programming, debugging, and system design.
Be concise and practical.
Rules:
- Use Markdown formatting
- ALWAYS wrap code in triple backticks with language
- Keep answers structured and readable
"""

def run_agent():
    print("\nType 'exit' to quit.\n")

    state = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT)]
    }

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Bye bro 👋")
            break

        state["messages"].append(HumanMessage(content=user_input))

        result = graph.invoke(state)

        state = result  # update state

        print(f"AI: {state['messages'][-1].content}")



def run_agent_once(user_input: str, history: list[str] | None = None) -> str:
    if history is None:
        history = []

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for msg in history:
        messages.append(HumanMessage(content=msg))

    messages.append(HumanMessage(content=user_input))

    state = {"messages": messages}

    result = graph.invoke(state)

    return result["messages"][-1].content

if __name__ == "__main__":
    run_agent()
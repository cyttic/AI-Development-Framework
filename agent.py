import os, json, copy
from typing import Any
from pathlib import Path
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

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

SYSTEM_PROMPT = """
You are an AI coding assistant inside a development framework.
Your goal is to help the user with programming, debugging, and system design.
Be concise and practical.
"""

def run_agent():
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    tracer = ToolTracer()

    print("\nType 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Bye bro 👋")
            break

        # Add user message
        messages.append(HumanMessage(content=user_input))

        # Call LLM
        response = llm_chat(messages)

        # Save response
        messages.append(response)

        # Print result
        print(f"AI: {response.content}")

if __name__ == "__main__":
    run_agent()
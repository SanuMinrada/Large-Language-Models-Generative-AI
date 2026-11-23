"""Shared tool-calling agent loop used by every agent script in Lab 6.

A minimal single-step tool-calling pattern: bind tools to the LLM, let it
decide whether to call one, run it, feed the result back, get the final
answer. Kept generic so each part (calculator-only, multi-tool, memory,
RAG) just supplies a different tool list and system prompt.
"""
import sys

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# Some tool results (e.g. RAG chunks pulled from lecture PDFs) contain
# Unicode characters like "→" that the default Windows console codepage
# can't encode, which crashes a plain print(). Reconfigure stdout to UTF-8
# with lossy fallback instead of hard-crashing on narrow terminals.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

LLM_MODEL = "llama3.2:3b"


def build_llm():
    return ChatOllama(model=LLM_MODEL, temperature=0, num_ctx=8192)


def run_turn(llm_with_tools, tools_by_name, history, user_input, verbose=True):
    """Run one user turn through the agent, executing any tool calls, and
    return the final text answer. `history` is a list of messages that is
    mutated in place to carry conversation context across turns."""
    history.append(HumanMessage(user_input))

    ai_msg = llm_with_tools.invoke(history)
    history.append(ai_msg)

    if not ai_msg.tool_calls:
        return ai_msg.content

    for call in ai_msg.tool_calls:
        tool = tools_by_name[call["name"]]
        if verbose:
            print(f"[TOOL SELECTED] {call['name']}")
            print(f"[TOOL ARGUMENTS] {call['args']}")
        result = tool.invoke(call["args"])
        if verbose:
            print(f"[TOOL RESULT] {result}")
        history.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    final = llm_with_tools.invoke(history)
    history.append(final)
    return final.content


def make_tools_by_name(tools):
    return {t.name: t for t in tools}


def chat_loop(system_prompt, tools, banner, on_user_message=None):
    """`on_user_message`, if given, is called with (llm, user_input) before
    each turn — used by Parts B/C to run memory extraction/saving."""
    llm = build_llm()
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = make_tools_by_name(tools)
    history = [SystemMessage(system_prompt)]

    print("=" * 40)
    print(banner)
    print("=" * 40)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        if on_user_message is not None:
            on_user_message(llm, user_input)

        answer = run_turn(llm_with_tools, tools_by_name, history, user_input)
        print(f"\nAgent: {answer}")

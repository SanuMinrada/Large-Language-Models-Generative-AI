"""Lab 6 - Part A: Building a Calculator Agent.

Run and ask things like "What is 25 times 4?" — the agent calls the
calculator tool, which opens the real Windows Calculator app, types the
expression, and returns the result.
"""
from agent_core import chat_loop
from tools_gui_calculator import gui_calculator

SYSTEM_PROMPT = (
    "You are a calculator assistant. For any arithmetic question, use the "
    "gui_calculator tool rather than computing it yourself."
)

if __name__ == "__main__":
    chat_loop(SYSTEM_PROMPT, [gui_calculator], "CALCULATOR AGENT")

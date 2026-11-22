"""Lab 6 - Part B: Give your AI Agent Memory.

Two tools (Calculator, Memory Search) plus a long-term SQLite memory store.
Try: "My name is Alex." then, in a later turn, "What is my name?"
"""
from agent_core import chat_loop
from tools_basic import calculator
from tools_memory import search_student_memory, extract_and_save_memory

SYSTEM_PROMPT = (
    "You are a helpful assistant with a calculator tool and a memory search tool "
    "(search_student_memory) for recalling information the user has told you "
    "earlier. Use search_student_memory whenever the user asks about something "
    "they may have told you before (e.g. their name)."
)

if __name__ == "__main__":
    chat_loop(
        SYSTEM_PROMPT,
        [calculator, search_student_memory],
        "MEMORY-ENABLED AGENT",
        on_user_message=extract_and_save_memory,
    )

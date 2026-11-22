"""Lab 6 - Exercise: Expand with Multiple Tools.

Calculator (pure computation here, not the GUI version — see Part A for
that), date/time, Notepad, file search, file creation, and system info.
"""
from agent_core import chat_loop
from tools_basic import BASIC_TOOLS_WITH_NOTEPAD

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools: calculator, get_datetime, "
    "open_notepad_and_write, search_files, create_file, and system_info. Use the "
    "appropriate tool whenever the user's request matches one, instead of guessing."
)

if __name__ == "__main__":
    chat_loop(SYSTEM_PROMPT, BASIC_TOOLS_WITH_NOTEPAD, "MULTI-TOOL AGENT")

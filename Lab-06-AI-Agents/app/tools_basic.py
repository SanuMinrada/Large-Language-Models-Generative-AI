"""Lab 6 - Part A / Exercise: calculator + the five "expand with multiple
tools" tools (date/time, Notepad, file search, file creation, system info).
"""
import datetime
import platform
import subprocess
import time
from pathlib import Path

import psutil
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '25 * 4', and return the result."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_datetime() -> str:
    """Return the current date and time."""
    return datetime.datetime.now().strftime("%A, %d %B %Y, %H:%M:%S")


@tool
def open_notepad_and_write(text: str) -> str:
    """Open Windows Notepad and type the given text into it."""
    import pyautogui
    import pygetwindow as gw

    subprocess.Popen(["notepad.exe"])
    time.sleep(1.5)

    windows = [w for w in gw.getWindowsWithTitle("Notepad") if w.title]
    if windows:
        windows[0].activate()
        time.sleep(0.3)

    pyautogui.typewrite(text, interval=0.02)
    return f"Opened Notepad and wrote: {text!r}"


@tool
def search_files(directory: str, pattern: str = "*") -> str:
    """Search for files by name/extension pattern (e.g. '*.pdf') in a directory."""
    base = Path(directory)
    if not base.exists():
        return f"Directory not found: {directory}"

    matches = [str(p) for p in base.rglob(pattern) if p.is_file()]
    if not matches:
        return f"No files matching '{pattern}' found in {directory}"
    return "\n".join(matches[:50])


@tool
def create_file(directory: str, filename: str, content: str) -> str:
    """Create a text file with the given content in the given directory."""
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    path = base / filename
    path.write_text(content, encoding="utf-8")
    return f"Created file: {path}"


@tool
def system_info() -> str:
    """Return operating system, processor, and memory information."""
    mem = psutil.virtual_memory()
    lines = [
        f"OS: {platform.system()} {platform.release()} ({platform.version()})",
        f"Processor: {platform.processor()}",
        f"CPU cores (logical): {psutil.cpu_count(logical=True)}",
        f"Total memory: {mem.total / (1024**3):.2f} GB",
        f"Available memory: {mem.available / (1024**3):.2f} GB",
    ]
    return "\n".join(lines)


BASIC_TOOLS = [calculator, get_datetime, search_files, create_file, system_info]
BASIC_TOOLS_WITH_NOTEPAD = BASIC_TOOLS + [open_notepad_and_write]

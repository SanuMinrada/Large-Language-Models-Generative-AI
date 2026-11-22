"""Lab 6 - Part A: a calculator tool that actually drives the Windows
Calculator app via pyautogui/pygetwindow, per the lab sheet's explicit
"Calculator should open and automatically perform calculations" requirement.

The numeric result is computed in Python (eval) as the reliable source of
truth for the agent's answer; the GUI automation is the visible demo of
"the agent controlling a real application" that the lab asks for.
"""
import subprocess
import time

from langchain_core.tools import tool

# Windows Calculator's standard view accepts digits/operators as direct
# keyboard input, so no on-screen button clicking is needed.
_KEY_MAP = {"+": "+", "-": "-", "*": "*", "/": "/", "(": "(", ")": ")"}


@tool
def gui_calculator(expression: str) -> str:
    """Open the Windows Calculator app, type the given arithmetic expression
    into it, press Enter, and return the computed result."""
    import pyautogui
    import pygetwindow as gw

    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except Exception as e:
        return f"Error evaluating expression: {e}"

    subprocess.Popen(["calc.exe"])
    time.sleep(1.5)

    windows = [w for w in gw.getWindowsWithTitle("Calculator") if w.title]
    if windows:
        windows[0].activate()
        time.sleep(0.3)

    pyautogui.typewrite(str(expression), interval=0.05)
    pyautogui.press("enter")

    return f"The result of the calculation is: {result}"

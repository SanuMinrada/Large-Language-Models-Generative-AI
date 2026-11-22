"""Lab 6 - Part D: Build with more tools (Google Calendar).

Requires your own credentials.json next to this file — see this Lab's
README for the Google Cloud / OAuth setup steps. Not runnable without it,
so this script was not executed during development (import-checked only).
"""
from agent_core import chat_loop
from tools_basic import calculator
from tools_calendar import check_todays_schedule, create_calendar_event

SYSTEM_PROMPT = (
    "You are a helpful assistant with a calculator and Google Calendar tools "
    "(check_todays_schedule, create_calendar_event). Use create_calendar_event "
    "only once you have a clear event summary, start time, and end time."
)

if __name__ == "__main__":
    chat_loop(
        SYSTEM_PROMPT,
        [calculator, check_todays_schedule, create_calendar_event],
        "CALENDAR AGENT",
    )

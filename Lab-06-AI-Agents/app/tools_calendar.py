"""Lab 6 - Part D: Google Calendar tools (create event, check today's schedule).

NOT independently testable by an assistant: Google Calendar access requires
an actual Google Cloud project, an OAuth 2.0 Desktop client, and the first
run opens a real browser for the account owner to grant consent — none of
which can be done without the portfolio owner's own Google account. See
this Lab's README for the manual setup steps (mirrors the lab sheet's own
screenshots: create a Cloud project, enable the Calendar API, create an
OAuth client, add yourself as a test user, download `credentials.json`).

Once `credentials.json` is present next to this file, running any of these
tools for the first time will open a browser for one-time consent and then
cache a `token.json` for subsequent runs.
"""
import datetime
from pathlib import Path

from langchain_core.tools import tool

SCOPES = ["https://www.googleapis.com/auth/calendar"]
CREDENTIALS_PATH = Path(__file__).parent / "credentials.json"
TOKEN_PATH = Path(__file__).parent / "token.json"


def _get_calendar_service():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    if not CREDENTIALS_PATH.exists():
        raise FileNotFoundError(
            "credentials.json not found — see this Lab's README for the Google "
            "Cloud / OAuth setup steps (this requires your own Google account)."
        )

    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)  # opens a browser for one-time consent
        TOKEN_PATH.write_text(creds.to_json())

    return build("calendar", "v3", credentials=creds)


@tool
def check_todays_schedule() -> str:
    """Check Google Calendar for today's events."""
    service = _get_calendar_service()

    now = datetime.datetime.now(datetime.timezone.utc)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    end_of_day = now.replace(hour=23, minute=59, second=59, microsecond=0).isoformat()

    events_result = service.events().list(
        calendarId="primary",
        timeMin=start_of_day,
        timeMax=end_of_day,
        singleEvents=True,
        orderBy="startTime",
    ).execute()

    events = events_result.get("items", [])
    if not events:
        return "No events scheduled for today."

    lines = []
    for event in events:
        start = event["start"].get("dateTime", event["start"].get("date"))
        lines.append(f"{start}: {event.get('summary', '(no title)')}")
    return "\n".join(lines)


@tool
def create_calendar_event(summary: str, start_time: str, end_time: str) -> str:
    """Create a new Google Calendar event. start_time/end_time must be ISO 8601
    datetimes, e.g. '2026-11-25T14:00:00+05:30'."""
    service = _get_calendar_service()

    event = {
        "summary": summary,
        "start": {"dateTime": start_time},
        "end": {"dateTime": end_time},
    }
    created = service.events().insert(calendarId="primary", body=event).execute()
    return f"Created event '{summary}' ({start_time} - {end_time}): {created.get('htmlLink')}"

"""Lab 6 - Part B: give the agent long-term memory via SQLite.

Short-term memory is just the in-process conversation history (handled by
agent_core's `history` list). Long-term memory is a small SQLite table of
facts extracted from the conversation, searchable by a dedicated tool.
"""
import sqlite3
from pathlib import Path

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

DB_PATH = Path(__file__).parent / "agent_memory.db"

EXTRACTION_PROMPT = """Does the following user message contain a durable fact worth
remembering about the user (e.g. their name, interests, course, preferences)?

Message: "{message}"

If yes, reply with ONLY a short sentence stating the fact (e.g. "The user's name is Alex.").
If no, reply with exactly: NONE"""


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, content TEXT)"
    )
    conn.commit()
    conn.close()


def save_memory(content: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO memories (content) VALUES (?)", (content,))
    conn.commit()
    conn.close()


def extract_and_save_memory(llm, user_message: str) -> str | None:
    """Ask the LLM whether this message contains a fact worth saving, and
    save it to SQLite if so. Returns the saved fact, or None."""
    response = llm.invoke([HumanMessage(EXTRACTION_PROMPT.format(message=user_message))])
    fact = response.content.strip()
    if fact and fact.upper() != "NONE":
        save_memory(fact)
        print(f"[MEMORY] Saved: {fact}")
        return fact
    print("[MEMORY] No new information to save.")
    return None


@tool
def search_student_memory(query: str) -> str:
    """Search previously stored information about the user/student."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT content FROM memories").fetchall()
    conn.close()

    if not rows:
        return "No matching memories found."

    query_words = set(query.lower().split())
    matches = [r[0] for r in rows if query_words & set(r[0].lower().split())]
    if not matches:
        # Fall back to returning everything stored — still useful context
        # for small memory stores like this one.
        return "\n".join(r[0] for r in rows)
    return "\n".join(matches)


init_db()

# Lab 06 — AI Agents

## Laboratory Session

Lab 06 — Building a Tool-Using AI Agent with Ollama

## Objective

Build a tool-calling AI agent on top of a local LLM (Ollama + `llama3.2:3b`), progressively
adding capability: a single calculator tool that drives the real Windows Calculator app,
multiple utility tools, long-term memory via SQLite, and a RAG tool reusing Lab 5's knowledge
base — per [Lab sheet 6](resources/Lab%20sheet%206.pdf). **Part E (n8n) and the final n8n-based
exercise were skipped per instruction**; Part D (Google Calendar) is implemented but requires
the portfolio owner's own Google OAuth credentials to actually run.

## Tasks and Implementation

Source: [app/](app/)

1. **[app/agent_core.py](app/agent_core.py):** a shared tool-calling loop (bind tools to
   `ChatOllama`, execute whichever tool the model calls, feed the result back, get the final
   answer) reused by every agent script below, so the tool-calling mechanics aren't duplicated
   five times over.
2. **Part A — [app/agent_calculator.py](app/agent_calculator.py) +
   [app/tools_gui_calculator.py](app/tools_gui_calculator.py):** a single calculator tool that
   actually opens the real Windows Calculator app via `pyautogui`/`pygetwindow`, types the
   expression, and returns the result (computed in Python as the reliable source of truth,
   alongside the visible GUI demonstration).
3. **Exercise — [app/agent_multi_tool.py](app/agent_multi_tool.py) +
   [app/tools_basic.py](app/tools_basic.py):** calculator, date/time, Notepad (opens and types
   text), file search, file creation, and system info (OS/CPU/memory via `psutil`) — all six
   tools from the exercise, tested against the lab sheet's own 6-query test table.
4. **Part B — [app/agent_with_memory.py](app/agent_with_memory.py) +
   [app/tools_memory.py](app/tools_memory.py):** a SQLite-backed long-term memory store. After
   each user message, a lightweight LLM call decides whether it contains a durable fact worth
   saving (e.g. the user's name); a `search_student_memory` tool lets the agent recall it later.
5. **Part C — [app/agent_rag.py](app/agent_rag.py) +
   [app/tools_rag.py](app/tools_rag.py):** adds a `search_university_knowledge` tool that
   queries **Lab 5's own persisted ChromaDB index directly** (imports `rag_core.py` from
   `../Lab-05-RAG-Pipeline/app/` rather than rebuilding RAG from scratch), per the lab sheet's
   "[Hint: Use the RAG we implemented Lab session 5]".
6. **Part D — [app/agent_calendar.py](app/agent_calendar.py) +
   [app/tools_calendar.py](app/tools_calendar.py):** Google Calendar tools
   (`check_todays_schedule`, `create_calendar_event`) using `google-api-python-client` +
   OAuth. **Not independently runnable** — see Observations below.

**To run locally:** reuses the same `rag_env` virtual environment from Lab 5 (install
`app/requirements.txt` into it), then run any of `agent_calculator.py`, `agent_multi_tool.py`,
`agent_with_memory.py`, or `agent_rag.py` directly. `agent_rag.py` requires Lab 5's `ingest.py`
to have been run first (needs its `chroma_db/` to exist).

## Results and Evidence

Real, tested output from each script (Unicode-containing tool results like `→` needed a small
stdout-encoding fix in `agent_core.py` to print correctly on the Windows console):

**Part A** — asked "What is 25 times 4?": the real Calculator app opened on screen (confirmed via
`Get-Process CalculatorApp`), and the agent replied "The result of 25 times 4 is 100."

**Exercise** — tested against the lab sheet's own query table:
```
Calculate 25 × 4                          -> 100 (calculator tool)
What is the date today?                   -> Saturday, 21 November 2026 (get_datetime tool)
Open Notepad and write Hello Students     -> Notepad opened with "Hello Students" typed in
                                              (confirmed via window title "*Hello Students - Notepad")
Create test.txt with Hello World          -> file created and verified on disk
Find all txt files in test_files          -> correctly found and listed test.txt
Show my computer specifications           -> OS: Windows 11, 8 logical cores, ~7.8 GB RAM
```

**Part B** — exact replication of the lab sheet's own memory demo:
```
You: My name is Alex.
[MEMORY] Saved: The user's name is Alex.
Agent: Hello Alex! It's nice to meet you. Is there something I can help you with today?

You: What is my name?
[TOOL SELECTED] search_student_memory  ->  "The user's name is Alex."
Agent: You already told me your name is Alex.
```

**Part C** — asked "What is NLP?": correctly retrieved real chunks from Lab 5's ingested
`Week 1.pdf` (via `search_university_knowledge`) and gave an accurate, grounded answer; a
follow-up "What is 12 times 8?" correctly used the calculator tool instead (96).

**Part D** — code is written and import-checked, but not run: Google Calendar access needs a
real Google Cloud project, an OAuth Desktop client, and a one-time interactive browser consent
flow under the portfolio owner's own Google account — none of which an assistant can complete on
someone else's behalf. See Observations.

*(Screenshots to be added here.)*

## Observations

- **A genuine tool-calling reliability limit:** `llama3.2:3b` reliably used proper structured
  tool calls for simple arguments (plain expressions, short relative paths), but for
  `create_file`/`search_files` with a long absolute Windows path (with escaped backslashes), it
  sometimes emitted a tool-call-shaped JSON string as plain *text* instead of a real tool call —
  the call never actually ran. Switching to short relative paths (e.g. `test_files` instead of a
  full `C:\Users\...` path) fixed it reliably. This is a real, reproducible limitation of a small
  3B local model's function-calling, not a bug in the tool code.
- A genuine bug caught during testing: printing a RAG tool result containing a `→` character
  crashed with `UnicodeEncodeError` on the default Windows console codepage. Fixed by
  reconfiguring `sys.stdout` to UTF-8 with `errors="replace"` in `agent_core.py` — worth knowing
  about for anyone building a CLI agent on Windows that handles arbitrary document text.
- Reusing Lab 5's `rag_core.py`/`chroma_db` directly (rather than re-implementing RAG for this
  lab) worked cleanly via a relative import — a concrete demonstration of why Lab 5's shared
  `rag_core.py` module was worth extracting in the first place.
- Part D is a good illustration of a hard limit on what can be fully tested end-to-end without
  the account owner personally present: OAuth consent for a real Google account cannot be
  completed by anyone other than that account's owner, no matter how correct the code is.

## Learning Outcomes

- Built a working tool-calling agent loop from scratch on a local LLM (no OpenAI/Anthropic API),
  and saw directly how LangChain's `bind_tools` + `ChatOllama` integration works.
- Learned a concrete limitation of small local models' function-calling: argument complexity
  (long paths with escaped characters) can silently cause the model to "describe" a tool call in
  text instead of actually making one — worth testing for, not just assuming tool-calling always
  works.
- Implemented a simple but genuine long-term memory system (SQLite + an LLM-driven extraction
  step), not just a conversation buffer — and confirmed it reproduces the exact "remembers your
  name across turns" behaviour the lab sheet demonstrates.
- Practiced reusing a previous lab's module (Lab 5's `rag_core.py`) directly via a relative
  import rather than duplicating it — real evidence that factoring shared logic out pays off
  across, not just within, a single lab.
- Recognised which parts of a system genuinely require a human in the loop (Google OAuth
  consent) versus which can be fully automated.

## Manual steps required (Part D — Google Calendar)

To actually run `agent_calendar.py`, you'll need to do this yourself (mirrors the lab sheet's own
screenshots):

1. Create a project in the [Google Cloud Console](https://console.cloud.google.com/).
2. Enable the **Google Calendar API** for that project.
3. Create an **OAuth client ID** (Application type: *Desktop app*).
4. Add yourself as a **test user** under the OAuth consent screen.
5. Download the client secret as `credentials.json` and place it in
   `Lab-06-AI-Agents/app/credentials.json` (gitignored — never commit real credentials).
6. Run `python agent_calendar.py` — the first run opens a browser for one-time consent, then
   caches a `token.json` next to `credentials.json` for subsequent runs.

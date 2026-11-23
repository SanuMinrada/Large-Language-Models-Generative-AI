# Lab 06: Building a Tool-Using AI Agent with Ollama

## Laboratory Session
Lab 06, Building a Tool-Using AI Agent with Ollama (PUSL 3205)

## Objective
Build a tool-calling AI agent on a local LLM (Ollama + `llama3.2:3b`), adding capability step by
step: a calculator tool controlling the real Windows Calculator, several utility tools, long-term
memory, a RAG tool reusing Lab 5's knowledge base, and a Google Calendar tool via OAuth. Part E
(n8n) was skipped per instruction; every other part, including Part D, was run and tested for real
using the portfolio owner's own Google account.

## Tasks and Implementation
- Background notes: [notes.md](notes.md)
- [app/agent_core.py](app/agent_core.py): a shared tool-calling loop (bind tools to `ChatOllama`,
  run whichever tool the model picks, feed the result back, get the final answer), reused by every
  agent script below. `ChatOllama` is given `num_ctx=8192` (instead of Ollama's small default) so
  a multi-turn, multi-tool session doesn't silently lose earlier context — see Observations
- Part A — [app/agent_calculator.py](app/agent_calculator.py): a calculator tool that opens the
  real Windows Calculator, types the expression, and returns the result
- Exercise — [app/agent_multi_tool.py](app/agent_multi_tool.py): calculator, date/time, Notepad,
  file search, file creation, and system info tools, tested against the lab sheet's own query table
- Part B — [app/agent_with_memory.py](app/agent_with_memory.py): a SQLite-backed long-term memory
  — after each message, a small LLM call decides whether to save a durable fact (e.g. the user's
  name), recalled later via a `search_student_memory` tool
- Part C — [app/agent_rag.py](app/agent_rag.py): adds a `search_university_knowledge` tool that
  queries Lab 5's own ChromaDB index directly, reusing `rag_core.py` rather than rebuilding RAG
- Part D — [app/agent_calendar.py](app/agent_calendar.py): Google Calendar tools
  (`check_todays_schedule`, `create_calendar_event`) via OAuth — a real Google Cloud project and
  OAuth consent flow were set up and tested end-to-end (see Results and Evidence and the manual
  steps below)
- To run locally: reuse Lab 5's `rag_env` virtual environment (install `app/requirements.txt`),
  then run any of the agent scripts directly (`agent_rag.py` needs Lab 5's `ingest.py` to have
  been run first)

## Results and Evidence
Real, tested output from each script:

**Part A** — asked "What is 7 times 4?" and "What is (12 + 8) times 3?": the real Calculator app
opened both times and showed the correct results (28, then 60).

**Exercise** — tested against the lab sheet's own query table, in one session:
```
Calculate 25 x 4                          -> 100 (calculator tool)
What is the date today?                   -> Monday, 23 November 2026 (get_datetime tool)
Open Notepad and write Hello Students     -> Notepad opened and typed the text
Create test.txt with Hello World          -> file created and verified on disk
Find all PDF files in ..\resources        -> correctly found Lab sheet 6.pdf and Week 6.pdf
Show my computer specifications           -> OS: Windows 11, 8 logical cores, 7.79 GB RAM
```
(The lab sheet's own file-search example path doesn't exist on this machine, so a real folder —
`resources/`, which already has PDFs in it — was used instead.)

**Part B** — matches the lab sheet's own memory demo (own name used instead of "Alex"):
```
You: My name is Sanuthi
[MEMORY] Saved: The user's name is Sanuthi.
[TOOL SELECTED] search_student_memory  ->  "The user's name is Sanuthi."
Agent: Hello Sanuthi! It's nice to meet you. Is there something I can help you with today?

You: What is my name?
[TOOL SELECTED] search_student_memory  ->  "The user's name is Sanuthi."
Agent: Hello Sanuthi!
```
This recall happened within the same run, which proves the SQLite-backed tool itself works;
proving it survives a full restart (a separate, stopped-and-restarted process) wasn't captured.

**Part C** — asked "What is a foundation model?", then "What is 15 times 6?" and "What is my
name?" in the same session: all three tools were routed correctly
(`search_university_knowledge` → `calculator` → `search_student_memory`), and the calculator/memory
answers were exactly right (90, "Sanuthi"). The RAG answer about foundation models *read* as
correct, but the actual retrieved chunks (an unrelated exercise, a code snippet, a slide heading)
weren't a good match for the question — see Observations for why.

**Part D** — fully set up and tested with a real Google Cloud project and OAuth consent flow:
asked "What's on my schedule today?" (correctly reported no events), then "Create an event called
'LLM Study Session' tomorrow from 3pm to 4pm" — the event was created via the API and confirmed
visible in Google Calendar afterwards. See the manual steps below for how this was set up.

Screenshots: see [screenshots/](screenshots/). Not every screenshot on the lab sheet's checklist
was captured — a representative subset was chosen for each part:

| Part | File | What it shows |
|------|------|----------------|
| Part A | [calculator-tool-selection.png](screenshots/calculator-tool-selection.png) | Tool call + result in the terminal |
| Part A | [calculator-opened.png](screenshots/calculator-opened.png) | The real Calculator app showing the answer |
| Part A | [calculator-second-test.png](screenshots/calculator-second-test.png) | A second, multi-step calculation |
| Exercise | [multi-tool-tests.png](screenshots/multi-tool-tests.png) | Calculator, date, file search, and system info tests |
| Exercise | [notepad-test.png](screenshots/notepad-test.png) | Notepad opened and typed into |
| Exercise | [file-creation-terminal.png](screenshots/file-creation-terminal.png) | `create_file` tool call |
| Exercise | [file-creation-content.png](screenshots/file-creation-content.png) | The created file's real contents |
| Part B | [memory-saved.png](screenshots/memory-saved.png) | A fact being saved to SQLite |
| Part B | [memory-recalled.png](screenshots/memory-recalled.png) | The same fact recalled via the memory tool |
| Part C | [rag-final-answer.png](screenshots/rag-final-answer.png) | The RAG tool's final answer |
| Part C | [three-tools-routing.png](screenshots/three-tools-routing.png) | Calculator and memory tools routed correctly in one session |
| Part D | [check-schedule-and-create-event.png](screenshots/check-schedule-and-create-event.png) | Checking today's schedule, then creating a real event |
| Part D | [event-in-calendar.png](screenshots/event-in-calendar.png) | The created event, visible in Google Calendar |

## Observations
- A real tool-calling limit: `llama3.2:3b` handled simple arguments fine, but for a long absolute
  Windows path with escaped backslashes, it sometimes described a tool call as plain text instead
  of actually making one. Short relative paths fixed it reliably — a genuine limitation of a small
  3B model's function-calling, not a bug in the code.
- A real bug caught during testing: printing RAG text containing "→" crashed with
  `UnicodeEncodeError` on the Windows console. Fixed by setting stdout to UTF-8 in `agent_core.py`.
- Reusing Lab 5's `rag_core.py` directly (instead of rebuilding RAG) worked cleanly via a relative
  import — good evidence that pulling shared logic into its own module paid off.
- Found that Lab 5's vector store had accidentally accumulated 4 duplicate copies of `Week 4.pdf`
  from repeated Streamlit upload testing during Lab 5 — this made `search_university_knowledge`
  return the exact same chunk 3 times for one query. Fixed by deleting `chroma_db` and
  re-ingesting once, cleanly.
- Even after that cleanup, RAG answers in Part C weren't always well-grounded: these are lecture
  slide PDFs (sparse per-slide text, key points often in diagrams rather than extractable text),
  chunking doesn't respect slide boundaries, the embedding model (`qwen3-embedding:0.6b`) is small,
  and `search_university_knowledge` has no relevance threshold — it always returns its top-3
  nearest chunks even when none of them are a good match. The model's answer can still sound
  right, but that's often its own general knowledge filling the gap, not genuine grounding.
- `agent_core.py`'s `ChatOllama` wasn't given an explicit context window, so it used Ollama's small
  default. In a long multi-tool session (several RAG calls, each appending full raw chunks to the
  conversation history), this could silently truncate older context, including the system prompt.
  Fixed by setting `num_ctx=8192`.
- Part D needed real, manual setup under the portfolio owner's own Google account (Cloud project,
  enabled Calendar API, OAuth consent screen, a test user, a downloaded `credentials.json`) — none
  of which can be done by an assistant. Also hit a real "Error 403: access_denied" wall on first
  sign-in until the Google account was explicitly added as a test user, which in Google's current
  console UI lives under the **Audience** tab rather than the old "OAuth consent screen" page.

## Learning Outcomes
- Built a working tool-calling agent loop on a local LLM, and saw how LangChain's `bind_tools` +
  `ChatOllama` fit together.
- Learned a real limitation of small local models' function-calling: complex arguments can make
  the model "describe" a tool call in text instead of making one.
- Implemented a genuine long-term memory system (SQLite + an LLM-driven save step), not just a
  chat buffer, and confirmed it remembers facts across turns.
- Reused a previous lab's module directly via a relative import instead of duplicating it.
- Learned that having a vector database isn't enough for good RAG answers — chunking strategy,
  embedding quality, and a relevance threshold all matter, and a model can mask weak retrieval by
  falling back on its own general knowledge.
- Learned to manage an LLM's context window deliberately (`num_ctx`) once a session involves
  several tool calls, instead of relying on a provider's default.
- Set up and completed a real Google OAuth consent flow end-to-end, and learned to recognise which
  parts of a system genuinely need a human in the loop (OAuth consent, Cloud Console setup) versus
  what can be fully automated.

## Manual steps required (Part D — Google Calendar)
These steps have to be done by the account owner — they were completed for real to test Part D:
1. Create a project in the [Google Cloud Console](https://console.cloud.google.com/).
2. Enable the Google Calendar API for that project.
3. Set up the OAuth consent screen (External), then add your own Google account as a **test
   user** — in the current console UI this is under the **Audience** tab, not a separate "OAuth
   consent screen" page. Without this step, signing in fails with "Error 403: access_denied".
4. Create an OAuth client ID (Application type: Desktop app) under **Clients**.
5. Download the client secret and save it as `Lab-06-AI-Agents/app/credentials.json` — it must be
   the file itself at that exact path, not a folder containing it (gitignored either way — never
   commit real credentials).
6. Run `python agent_calendar.py` and ask it something — the first run opens a browser for
   one-time consent, then caches a `token.json` for subsequent runs.

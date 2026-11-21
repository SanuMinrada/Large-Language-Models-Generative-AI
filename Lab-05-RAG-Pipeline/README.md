# Lab 05 — RAG Pipeline

## Laboratory Session

Lab 05 — RAG Architecture (Retrieval-Augmented Generation)

## Objective

Set up a local RAG stack (Python + LangChain + Ollama + ChromaDB), build an ingestion script and
a retrieval/generation pipeline, and extend the app to read multiple PDFs with user-uploaded
documents — per [Lab sheet 5](resources/Lab%20sheet%205.pdf).

## Tasks and Implementation

Source: [app/](app/)

1. **Environment setup:** Python 3.12 virtual environment (`rag_env`), with LangChain,
   `langchain-community`, `langchain-ollama`, `langchain-chroma`, ChromaDB, Streamlit, PyMuPDF,
   `python-docx`, and `python-dotenv` installed (`sentence-transformers` was left out — it's not
   actually used by this implementation since embeddings are generated through Ollama, not
   loaded directly via `sentence-transformers`/PyTorch).
2. **LLM installation:** `ollama pull llama3.2:3b` (the LLM) and `ollama pull qwen3-embedding:0.6b`
   (the embedding model), both served locally via Ollama (`http://localhost:11434`).
3. **[app/rag_core.py](app/rag_core.py):** shared building blocks — PDF loading + chunking
   (`RecursiveCharacterTextSplitter`), Ollama embeddings, Chroma vector store (build/load), and
   the retrieve → prompt → LLM → parse chain — used by both entry points below so the logic isn't
   duplicated.
4. **[app/ingest.py](app/ingest.py):** loads every PDF in a `data/` folder (supports **multiple
   PDFs at once**), splits them into chunks, embeds them with the Qwen3 embedding model, and
   persists them into a ChromaDB collection.
5. **[app/rag_pipeline.py](app/rag_pipeline.py):** a terminal Q&A loop — retrieves the top-4
   relevant chunks for a question, grounds Llama 3.2 3B's answer in that context (with an explicit
   "say you don't know rather than guessing" instruction), matching the lab sheet's example
   session style.
6. **[app/streamlit_app.py](app/streamlit_app.py) — the Task requirement:** a "University AI
   Assistant" chat UI (matching the lab sheet's example screenshot) with a sidebar PDF uploader —
   users can upload their own PDF(s) at any time, which are ingested into the same vector store
   and immediately queryable, on top of whatever was already ingested via `ingest.py`.

Tested by ingesting two of this portfolio's own lecture PDFs as sample "university documents"
([Week 1](../Lab-01-Machine-Learning/resources/Week%201.pdf) and
[Week 4](../Lab-04-Fine-Tune/resources/Week%204.pdf)) — 79 chunks total — rather than shipping
duplicate PDF copies into this folder.

**To run locally:** create the venv and install `app/requirements.txt`, put PDFs in `app/data/`,
run `python ingest.py`, then either `python rag_pipeline.py` (terminal) or
`streamlit run streamlit_app.py` (browser UI). The resulting `chroma_db/` vector store and the
`rag_env/` virtual environment are intentionally excluded from version control (see
`.gitignore`) since both are fully regenerable from these scripts.

## Results and Evidence

Real, grounded Q&A output from `rag_pipeline.py` against the ingested Week 1 / Week 4 PDFs:

```
Ask a university question (or exit): What is NLP?

Answer: NLP stands for Natural Language Processing, which is a field at the intersection
of Artificial Intelligence and Linguistics that enables interaction between humans and
computers using natural language.

Ask a university question (or exit): What is LoRA?

Answer: LoRA stands for Low-Rank Adaptation. It is a technique that reduces the number of
parameters that need to be trained by creating a small subset of the base model to
fine-tune instead of adding layers to the model.
```

Both answers are accurate and correctly grounded in the ingested lecture content (cross-check
against [Lab-01's notes](../Lab-01-Machine-Learning/notes.md) and
[Lab-04's notes](../Lab-04-Fine-Tune/notes.md)) — confirming retrieval is pulling the right
chunks and the LLM is using them rather than its own general knowledge.

The Streamlit app was smoke-tested (boots cleanly, serves on `localhost`, no import/runtime
errors) but not screenshotted here — browser UI screenshots are easier to capture by actually
running it interactively.

*(Screenshots to be added here.)*

## Observations

- Asked a question whose answer genuinely isn't well-covered by the ingested PDFs (e.g. "What is
  a transformer block?" when only tangential mentions exist) and the model partially followed the
  "say you don't know" instruction, though its phrasing was a little self-contradictory — a
  reminder that prompt-level grounding instructions help but don't fully eliminate odd behaviour
  from a 3B model.
- Separating the shared logic into `rag_core.py` (used by both the CLI script and the Streamlit
  app) avoided duplicating the retrieval/prompt/chain code — the Streamlit upload feature reuses
  exactly the same `build_vectorstore`/`build_chain` functions as the command-line ingestion path.
- Installing `sentence-transformers` on Windows hit a path-length error from one of PyTorch's
  deeply nested header files — since this pipeline's embeddings actually go through Ollama (not
  `sentence-transformers` directly), it was simplest to drop that one dependency rather than work
  around the Windows path limit.

## Learning Outcomes

- Set up a fully local RAG stack: Ollama for both the LLM and embeddings, ChromaDB as the vector
  store, LangChain to wire retrieval and generation together.
- Learned the practical shape of a RAG pipeline end-to-end: load → chunk → embed → store →
  retrieve → ground a prompt → generate — not just as a diagram, but as working, testable code.
- Extended a basic single-document RAG app into one that supports multiple PDFs and live user
  uploads, without duplicating the core retrieval logic.
- Saw directly why grounding prompts ("answer only from this context, say you don't know
  otherwise") matter, and that they reduce but don't eliminate ungrounded or inconsistent answers
  from a small local model.

# Week 5 Reflection — RAG Pipeline

## What I learned

This lab made RAG feel like a real system instead of just a diagram. Setting up Ollama, ChromaDB,
and LangChain locally and actually watching a question get answered using real chunks from a PDF
was satisfying — I could see retrieval and generation working together instead of just reading
about it. Keeping the core logic in `rag_core.py` and reusing it for both the terminal script and
the Streamlit app also felt like a good decision, since the upload feature didn't need any extra
retrieval code of its own.

## What was challenging

The biggest lesson came later, while reusing this same database in Lab 6. I found out my vector
store had the same PDF saved into it multiple times because I'd uploaded it again and again while
testing the Streamlit app, without ever clearing the old data out first. That one mistake made
retrieval return the same chunk repeatedly instead of actually useful results. It taught me that
just having a vector database isn't enough — what's actually inside it matters a lot, and it's
easy to pollute it without noticing.

## What I'd try next

- Keep a cleaner habit of resetting the database before re-testing uploads, instead of letting it
  grow with duplicates.
- Try smaller chunks, since these PDFs are lecture slides and a chunk can easily mix parts of
  different slides together.
- Add a relevance score check, so the app can say "I don't know" instead of confidently using
  chunks that aren't actually relevant to the question.

"""Lab 6 - Part C: RAG Search Tool, reusing Lab 5's RAG implementation
directly (per the lab sheet's hint) rather than rebuilding it.
"""
import sys
from pathlib import Path

from langchain_core.tools import tool

# Lab 5's app/ lives as a sibling lab folder in the final repo layout:
# Lab-06-AI-Agents/app/tools_rag.py -> ../../Lab-05-RAG-Pipeline/app
_CANDIDATES = [
    Path(__file__).parent.parent.parent / "Lab-05-RAG-Pipeline" / "app",
    Path(__file__).parent.parent / "rag_lab",  # local/dev fallback
]
for candidate in _CANDIDATES:
    if (candidate / "rag_core.py").exists():
        sys.path.insert(0, str(candidate))
        _RAG_APP_DIR = candidate
        break
else:
    _RAG_APP_DIR = None

if _RAG_APP_DIR is not None:
    from rag_core import load_vectorstore  # noqa: E402

    PERSIST_DIR = _RAG_APP_DIR / "chroma_db"
    COLLECTION_NAME = "university_docs"
    _vectordb = load_vectorstore(COLLECTION_NAME, PERSIST_DIR) if PERSIST_DIR.exists() else None
else:
    _vectordb = None


@tool
def search_university_knowledge(query: str) -> str:
    """Retrieve relevant information from the university knowledge base (Lab 5's RAG index)."""
    if _vectordb is None:
        return "RAG index not found — run Lab 5's ingest.py first."

    docs = _vectordb.similarity_search(query, k=3)
    if not docs:
        return "No relevant information found."

    chunks = []
    for doc in docs:
        source = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page", "?")
        chunks.append(f"Source: {source}, Page: {page}\n{doc.page_content}")
    return "\n\n".join(chunks)

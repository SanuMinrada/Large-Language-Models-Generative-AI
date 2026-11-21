"""Lab Sheet 5 - RAG architecture: retrieval + generation pipeline (CLI).

Retrieves the most relevant chunks from the ChromaDB vector store built by
ingest.py, and uses Llama 3.2 3B (via Ollama) to answer the user's question
grounded in that retrieved context.
"""
from pathlib import Path

from rag_core import load_vectorstore, build_chain

PERSIST_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "university_docs"


def main():
    if not PERSIST_DIR.exists():
        print("No vector store found — run ingest.py first.")
        return

    vectordb = load_vectorstore(COLLECTION_NAME, PERSIST_DIR)
    chain = build_chain(vectordb)
    print("University RAG assistant ready. Type 'exit' to quit.\n")

    while True:
        question = input("Ask a university question (or exit): ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        answer = chain.invoke(question)
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()

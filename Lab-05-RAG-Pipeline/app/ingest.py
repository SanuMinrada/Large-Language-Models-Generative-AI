"""Lab Sheet 5 - RAG architecture: ingestion script.

Loads every PDF in DATA_DIR, splits it into chunks, embeds the chunks with
the Qwen3 embedding model served by Ollama, and persists them into a
ChromaDB vector store for the RAG pipeline (rag_pipeline.py) to query.

Supports reading multiple PDFs at once (the Lab 5 task requirement) — just
drop any number of PDF files into data/ before running this script.
"""
import sys
from pathlib import Path

from rag_core import load_and_split_pdfs, build_vectorstore

DATA_DIR = Path(__file__).parent / "data"
PERSIST_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "university_docs"


def main():
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {DATA_DIR}")
        sys.exit(1)

    print(f"Found {len(pdf_paths)} PDF(s): {', '.join(p.name for p in pdf_paths)}")
    chunks = load_and_split_pdfs(pdf_paths)
    print(f"Split into {len(chunks)} chunks.")

    build_vectorstore(chunks, COLLECTION_NAME, PERSIST_DIR)
    print(f"Ingested {len(chunks)} chunks into ChromaDB at {PERSIST_DIR}")


if __name__ == "__main__":
    main()

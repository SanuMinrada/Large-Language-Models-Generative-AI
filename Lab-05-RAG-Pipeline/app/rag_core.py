"""Shared RAG building blocks used by both rag_pipeline.py (CLI) and
streamlit_app.py (web UI with PDF upload) — avoids duplicating the
ingestion/retrieval/generation logic between the two entry points.
"""
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL = "llama3.2:3b"

PROMPT_TEMPLATE = """You are a helpful university assistant. Answer the question using
ONLY the context below. If the answer is not in the context, say you don't know
rather than guessing.

Context:
{context}

Question: {question}

Answer:"""


def load_and_split_pdfs(pdf_paths, chunk_size=800, chunk_overlap=120):
    """Load one or more PDF files and split them into chunks."""
    docs = []
    for path in pdf_paths:
        docs.extend(PyMuPDFLoader(str(path)).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def build_vectorstore(chunks, collection_name, persist_directory):
    """Embed chunks and persist them into a (new or existing) Chroma collection."""
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )


def load_vectorstore(collection_name, persist_directory):
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_directory),
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(vectordb, k=4):
    """Build the retrieve -> prompt -> LLM -> parse chain for a given vector store."""
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

"""Lab Sheet 5 - RAG architecture: Streamlit chat UI.

Task requirement: let the app read more than one PDF, and let users upload
their own PDFs and get answers from them — this app supports both: it
starts from whatever's already ingested into data/ (via ingest.py), and
lets the user add more PDFs from the browser at any time.
"""
import tempfile
from pathlib import Path

import streamlit as st

from rag_core import load_and_split_pdfs, build_vectorstore, load_vectorstore, build_chain

PERSIST_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "university_docs"

st.set_page_config(page_title="University AI Assistant", page_icon="🎓")
st.title("🎓 University AI Assistant")
st.caption("Ask questions based on university documents")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectordb" not in st.session_state:
    st.session_state.vectordb = (
        load_vectorstore(COLLECTION_NAME, PERSIST_DIR) if PERSIST_DIR.exists() else None
    )

with st.sidebar:
    st.header("Upload documents")
    uploaded_files = st.file_uploader(
        "Upload your own PDF(s)", type="pdf", accept_multiple_files=True
    )
    if uploaded_files and st.button("Ingest uploaded PDFs"):
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_paths = []
            for uploaded in uploaded_files:
                path = Path(tmpdir) / uploaded.name
                path.write_bytes(uploaded.getvalue())
                saved_paths.append(path)

            with st.spinner(f"Reading {len(saved_paths)} PDF(s) and updating the index..."):
                chunks = load_and_split_pdfs(saved_paths)
                st.session_state.vectordb = build_vectorstore(
                    chunks, COLLECTION_NAME, PERSIST_DIR
                )
        st.success(f"Ingested {len(uploaded_files)} new PDF(s). You can ask about them now.")

    if st.session_state.vectordb is None:
        st.info("No documents indexed yet — upload a PDF to get started.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask your university question..."):
    if st.session_state.vectordb is None:
        st.warning("Please upload at least one PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain = build_chain(st.session_state.vectordb)
                answer = chain.invoke(question)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

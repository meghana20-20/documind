"""
app.py
Streamlit front-end for DocuMind.
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from src.ingest import ingest_documents
from src.embed_store import VectorStore
from src.llm_client import LLMClient
from src.qa_engine import answer_question

load_dotenv()

st.set_page_config(page_title="DocuMind", page_icon="🧠", layout="wide")

if "store" not in st.session_state:
    st.session_state.store = VectorStore()
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🧠 DocuMind")
st.caption("RAG document Q&A with built-in hallucination detection")

with st.sidebar:
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Index documents"):
        with st.spinner("Chunking and embedding documents..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                paths = []
                for f in uploaded_files:
                    path = os.path.join(tmpdir, f.name)
                    with open(path, "wb") as out:
                        out.write(f.getbuffer())
                    paths.append(path)
                chunks = ingest_documents(paths)
                st.session_state.store.add(chunks)
                st.session_state.indexed_files.extend(f.name for f in uploaded_files)
        st.success(f"Indexed {len(uploaded_files)} document(s), {len(chunks)} chunks total.")

    if st.session_state.indexed_files:
        st.subheader("Indexed files")
        for name in st.session_state.indexed_files:
            st.write(f"• {name}")

    st.divider()
    st.subheader("⚙️ LLM Backend")
    st.write(f"Provider: `{os.getenv('LLM_PROVIDER', 'ollama')}`")
    st.caption("Set LLM_PROVIDER in your .env file to switch between openai / anthropic / ollama.")

st.divider()

question = st.text_input("Ask a question about your documents:")

if question:
    if not st.session_state.indexed_files:
        st.warning("Please upload and index at least one document first.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            llm = LLMClient()
            result = answer_question(question, st.session_state.store, llm)
            st.session_state.history.append((question, result))

        confidence_pct = result.overall_confidence * 100
        if confidence_pct >= 55:
            badge, color = "High confidence", "green"
        elif confidence_pct >= 40:
            badge, color = "Medium confidence", "orange"
        else:
            badge, color = "Low confidence — verify manually", "red"

        st.markdown(f"### Answer")
        st.write(result.answer)
        st.markdown(f":{color}[**{badge}** — groundedness score: {confidence_pct:.1f}%]")

        with st.expander("🔍 Sentence-level groundedness breakdown"):
            for s in result.sentence_scores:
                icon = "✅" if s.supported else "⚠️"
                st.write(f"{icon} `{s.similarity:.2f}` — {s.sentence}")

        with st.expander("📄 Source chunks used"):
            for chunk in result.sources:
                st.markdown(f"**{chunk.source}** (chunk {chunk.chunk_id})")
                st.text(chunk.text[:500] + ("..." if len(chunk.text) > 500 else ""))
                st.divider()

if st.session_state.history:
    st.divider()
    st.subheader("Conversation history")
    for q, r in reversed(st.session_state.history[:-1] if question else st.session_state.history):
        st.markdown(f"**Q:** {q}")
        st.write(r.answer)
        st.caption(f"Confidence: {r.overall_confidence * 100:.1f}%")
        st.divider()

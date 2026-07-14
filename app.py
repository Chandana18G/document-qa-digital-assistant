"""
app.py — a simple web chat UI built with Streamlit.

Usage:
    streamlit run app.py
Then open the URL it prints (usually http://localhost:8501).
Make sure you have run `python ingest.py` first so the index exists.
"""

from pathlib import Path
import streamlit as st

from rag import answer_question, INDEX_DIR

st.set_page_config(page_title="Document Q&A Chatbot (RAG)", page_icon="🤖")
st.title("📚 Document Q&A Chatbot")
st.caption("Retrieval-Augmented Generation — answers grounded in your own documents.")

if not (Path(INDEX_DIR) / "faiss.index").exists():
    st.warning("No index found. Run `python ingest.py` first, then reload this page.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

# Show past turns.
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input("Ask a question about your documents...")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            answer, sources = answer_question(question)
        reply = f"{answer}\n\n---\n*Sources: {', '.join(sources)}*"
        st.markdown(reply)
    st.session_state.history.append({"role": "assistant", "content": reply})

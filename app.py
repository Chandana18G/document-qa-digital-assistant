"""
app.py — a simple web chat UI built with Streamlit.

Usage:
    streamlit run app.py
Then open the URL it prints (usually http://localhost:8501).

Documents can be uploaded and indexed from the sidebar, or added to documents/
and indexed with `python ingest.py`.
"""

from datetime import datetime
from pathlib import Path
import streamlit as st

import rag

st.set_page_config(page_title="Document Q&A Chatbot (RAG)", page_icon="🤖")
st.title("📚 Document Q&A Chatbot")
st.caption("Retrieval-Augmented Generation — answers grounded in your own documents.")

if "turns" not in st.session_state:
    # One dict per answered question: question, answer, contexts, sources, search_query.
    st.session_state.turns = []
    st.session_state.upload_key = 0
    st.session_state.notice = None


def export_markdown() -> str:
    lines = [f"# Document Q&A chat — {datetime.now():%Y-%m-%d %H:%M}", ""]
    for t in st.session_state.turns:
        lines += [f"**You:** {t['question']}", "", f"**Assistant:** {t['answer']}", ""]
        for i, c in enumerate(t["contexts"], 1):
            lines.append(f"- [{i}] {c['label']}")
        lines.append("")
    return "\n".join(lines)


def chat_controls():
    """Clear/export buttons. Rendered last, so the export includes the newest answer."""
    with st.sidebar:
        st.header("Chat")
        col1, col2 = st.columns(2)
        if col1.button("Clear chat", use_container_width=True):
            st.session_state.turns = []
            st.rerun()
        col2.download_button("Export chat", data=export_markdown(), mime="text/markdown",
                             file_name=f"chat-{datetime.now():%Y%m%d-%H%M}.md",
                             disabled=not st.session_state.turns, use_container_width=True)


def show_details(turn):
    """Sources, the rewritten search query, and the numbered passages under an answer."""
    if turn["sources"]:
        st.markdown(f"---\n*Sources: {', '.join(turn['sources'])}*")
    if turn["search_query"] != turn["question"]:
        st.caption(f"Searched for: {turn['search_query']}")
    if turn["contexts"]:
        with st.expander(f"Show passages ({len(turn['contexts'])})"):
            for i, c in enumerate(turn["contexts"], 1):
                score = (f"re-rank score {c['rerank_score']:.1f}" if c["rerank_score"] is not None
                         else f"similarity {c['similarity']:.2f}")
                st.markdown(f"**[{i}] {c['label']}** · {score}")
                st.text(c["text"])


with st.sidebar:
    st.header("Documents")
    if st.session_state.notice:
        kind, message = st.session_state.notice
        getattr(st, kind)(message)
        st.session_state.notice = None
    uploads = st.file_uploader("Add PDF, Markdown or text files", type=["pdf", "md", "txt"],
                               accept_multiple_files=True, key=f"uploads_{st.session_state.upload_key}")
    if st.button("Add files & rebuild index" if uploads else "Rebuild index", use_container_width=True):
        rag.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploads or []:
            # Keep only the file name so an upload can't write outside documents/.
            (rag.DOCS_DIR / Path(f.name).name).write_bytes(f.getvalue())
        with st.spinner("Indexing documents..."):
            try:
                n_docs, n_chunks = rag.build_index()
                st.session_state.notice = ("success", f"Indexed {n_docs} document(s), {n_chunks} chunks.")
            except SystemExit as e:
                st.session_state.notice = ("error", str(e))
        st.session_state.upload_key += 1   # empties the uploader
        st.rerun()

    st.header("Settings")
    choice = st.selectbox("Answer model", ["auto"] + rag.PROVIDERS,
                          help="auto uses Claude or OpenAI if an API key is set, then Ollama "
                               "if it is running, else the local flan-t5 model.")
    provider = rag.get_provider(choice)
    if choice == "auto":
        st.caption(f"Using: {provider}")
    model = st.text_input("Model name", placeholder=rag.DEFAULT_MODELS[provider],
                          key=f"model_{provider}", help="Leave empty for the default.")
    temperature = st.slider("Temperature", 0.0, 1.0, rag.TEMPERATURE, 0.05,
                            help="Used by OpenAI and Ollama. Claude and the local model ignore it.")
    top_k = st.slider("Passages to use", 1, 10, rag.TOP_K)
    rerank = st.toggle("Re-rank passages", value=rag.RERANK,
                       help="Re-score candidates with a cross-encoder. More accurate, a bit slower.")
    if rerank:
        min_score = st.slider("Relevance threshold (re-rank score)", -12.0, 8.0, rag.MIN_RERANK_SCORE, 0.5,
                              help="Passages scoring below this are ignored. Off-topic text scores about -10.")
    else:
        min_score = st.slider("Relevance threshold (similarity)", 0.0, 1.0, rag.MIN_SIMILARITY, 0.05,
                              help="Passages less similar to the question than this are ignored.")


if not rag.index_exists():
    st.info("No documents indexed yet. Upload files in the sidebar and click "
            "**Add files & rebuild index**, or run `python ingest.py`.")
    chat_controls()
    st.stop()

# Show past turns.
for turn in st.session_state.turns:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        show_details(turn)

question = st.chat_input("Ask a question about your documents...")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        history = [(t["question"], t["answer"]) for t in st.session_state.turns]
        try:
            with st.spinner("Searching your documents..."):
                result = rag.answer_question(
                    question, history, stream=True, provider=choice, model=model or None,
                    temperature=temperature, top_k=top_k, rerank=rerank, min_score=min_score,
                )
            answer = st.write_stream(result["answer"])
            turn = {"question": question, "answer": answer.strip(), "contexts": result["contexts"],
                    "sources": result["sources"], "search_query": result["search_query"]}
            show_details(turn)
            st.session_state.turns.append(turn)
        except Exception as e:
            st.error(f"Could not answer: {e}")

chat_controls()

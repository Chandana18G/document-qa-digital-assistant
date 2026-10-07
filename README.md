# 📚 Document Q&A Digital Assistant — Knowledge Management (RAG)

A digital assistant that answers questions about **your own documents**
(PDF, Markdown, or text). It uses **retrieval-augmented generation**: the app
first finds the most relevant passages in your files, then asks a language model
to answer using *only* those passages — so answers stay grounded and avoid
hallucination.

Built with Python, Sentence-Transformers, FAISS, and Streamlit.

---

## Why this project

General language models don't know your private manuals, policies, or notes.
RAG lets a chatbot answer questions about that material without retraining the
model — when the documents change, you just re-index them. This is the pattern
most companies use for internal Generative AI assistants.

---

## Setup

Requires Python 3.9+.

```bash
git clone https://github.com/Chandana18G/document-qa-digital-assistant.git
cd document-qa-digital-assistant
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the embedding and re-ranking models (~90 MB each) and,
if no API key or Ollama model is available, `flan-t5-large` (~3 GB).

## Usage

1. **Add documents.** Put your `.pdf`, `.md` or `.txt` files in `documents/`
   (subfolders are fine), or upload them from the web app's sidebar. A `sample.md`
   is included so you can try it right away.
2. **Build the index.** Re-run this whenever your documents change (the web app
   has a **Rebuild index** button that does the same):
   ```bash
   python ingest.py
   ```
3. **Ask questions** from the command line…
   ```bash
   python ask.py "How many days per week can I work remotely?"
   python ask.py          # interactive mode with follow-up questions; empty line quits
   ```
   …or in the web chat:
   ```bash
   streamlit run app.py   # opens http://localhost:8501
   ```

### Web app features

- **Streaming answers** with inline citations like `[1]` (Claude, OpenAI, Ollama).
- **Show passages** under each answer: the numbered passages the citations refer to,
  with their source file, PDF page number and relevance score.
- **Sidebar:** upload documents and rebuild the index; choose the answer model, model
  name, temperature, number of passages, re-ranking and relevance threshold; clear the
  chat or export it as Markdown.

### Configuration (environment variables)

**Answer model.** By default (`LLM_PROVIDER=auto`) the app uses the first one available:

1. **Claude** (`claude-opus-5-5`) if `ANTHROPIC_API_KEY` is set
2. **OpenAI** (`gpt-4o-mini`) if `OPENAI_API_KEY` is set
3. **Ollama** if it is running locally and `OLLAMA_MODEL` is pulled — free and much
   better than flan-t5: install [Ollama](https://ollama.com), then `ollama pull llama3.2`
4. **flan-t5-large**, a small local model that needs no setup (short answers only)

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `auto` | Force a provider: `anthropic`, `openai`, `ollama` or `local`. |
| `ANTHROPIC_API_KEY` | *(unset)* | Enables Claude. `ANTHROPIC_MODEL` overrides the model. |
| `OPENAI_API_KEY` | *(unset)* | Enables OpenAI. `OPENAI_MODEL` overrides the model. |
| `OLLAMA_MODEL` / `OLLAMA_HOST` | `llama3.2` / `http://localhost:11434` | Ollama model and server. |
| `RAG_LOCAL_MODEL` | `google/flan-t5-large` | Local fallback model; `google/flan-t5-base` needs less memory. |
| `RAG_RERANK` | `1` | `0` turns off cross-encoder re-ranking (faster, less accurate). |
| `RAG_MIN_RERANK_SCORE` | `-4.0` | Relevance cut-off with re-ranking on; below it the app answers "couldn't find". |
| `RAG_MIN_SIMILARITY` | `0.35` | Relevance cut-off (cosine similarity) with re-ranking off. |
| `RAG_TEMPERATURE` | `0.1` | Sampling temperature for OpenAI and Ollama. |
| `RAG_DOCS_DIR` | `./documents` | Folder to read documents from. |
| `RAG_INDEX_DIR` | `./index` | Folder where the vector index is stored. |

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
```

---

## How it works

![Architecture](architecture.svg)

```
                ┌─────────────────────────────────────────────┐
   documents/   │  1. INGEST   read PDF / MD / TXT files        │
  (your files)  │  2. CHUNK    split into ~500-char passages    │
        │       │  3. EMBED    turn each chunk into a vector     │
        ▼       │              (all-MiniLM-L6-v2)                │
   ┌─────────┐  │  4. STORE    save vectors in a FAISS index     │
   │ ingest  │──┼──────────────────────────────────────────────┘
   │  .py    │        writes -> index/
   └─────────┘
                ┌─────────────────────────────────────────────┐
   your         │  5. REWRITE a follow-up into a standalone     │
  question ────▶│     question using the chat history           │
                │  6. RETRIEVE vector (FAISS) + keyword (BM25)  │
                │     search, re-rank, drop irrelevant chunks   │
                │  7. GENERATE an answer from those chunks (LLM) │
                └─────────────────────────────────────────────┘
                                   │
                                   ▼
                       grounded answer + sources
```

- **Chunking:** follows the document structure — chunks stay within one Markdown
  section, break between paragraphs or sentences, and start with their section
  title (e.g. `Remote Work Policy > Equipment`).
- **Embedding model:** `all-MiniLM-L6-v2` (small, fast, free).
- **Hybrid search:** FAISS cosine similarity plus BM25 keyword search (good at exact
  terms, names and numbers), merged with reciprocal rank fusion.
- **Re-ranking:** the `ms-marco-MiniLM-L-6-v2` cross-encoder scores each candidate
  against the question; chunks below the relevance threshold are dropped, and if
  none are left the app says it couldn't find the answer instead of guessing.
- **Answer model:** Claude, OpenAI, Ollama or a local flan-t5 model — it runs with
  **zero cost** out of the box. Hosted models and Ollama cite the passages they used.
- **Sources:** each passage is labelled with its path inside `documents/` and, for
  PDFs, its page number (e.g. `hr/handbook.pdf p. 4`).

## Project structure

| File | What it does |
|------|--------------|
| `rag.py` | Core engine: read, chunk, embed, retrieve, generate. |
| `ingest.py` | Builds the FAISS vector index from `documents/`. |
| `ask.py` | Command-line question answering. |
| `app.py` | Streamlit web chat interface. |
| `documents/` | Put your source files here (includes `sample.md`). |
| `architecture.svg` | Pipeline diagram. |
| `.streamlit/config.toml` | Streamlit settings (file watcher off — restart after editing `app.py`). |
| `requirements.txt` | Python dependencies. |

---

## What I learned / demonstrated

- Building a full RAG pipeline end to end (chunking, embeddings, vector search,
  prompt construction, grounded generation).
- Using FAISS for fast semantic similarity search.
- Prompt engineering to constrain the model to the retrieved context and reduce
  hallucination.
- Designing the code so the LLM backend is swappable (cloud API or local model).

---

## Possible next steps

- Swap FAISS for a hosted vector database (e.g. Chroma, Pinecone).
- Deploy the Streamlit app to the cloud.

---

*Built by Chandana Gurusiddappa — M.Sc. Data Science & AI.
[GitHub](https://github.com/Chandana18G) · [LinkedIn](https://linkedin.com/in/chandana-gurusiddappa-785563223)*

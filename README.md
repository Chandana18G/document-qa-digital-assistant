# 📚 Document Q&A Digital Assistant — Knowledge Management (RAG)

A digital assistant that answers questions about **your own documents**
(PDF, Markdown, or text). It uses **retrieval-augmented generation**: the app
first finds the most relevant passages in your files, then asks a language model
to answer using *only* those passages — so answers stay grounded and avoid
hallucination.

Built with Python, Sentence-Transformers, FAISS, and Streamlit.

![Architecture](architecture.svg)

---

## Why this project

General language models don't know your private manuals, policies, or notes.
RAG lets a chatbot answer questions about that material without retraining the
model — when the documents change, you just re-index them. This is the pattern
most companies use for internal Generative AI assistants.

---

## How it works

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
   your         │  5. EMBED the question the same way           │
  question ────▶│  6. RETRIEVE the top-k closest chunks (FAISS) │
                │  7. GENERATE an answer from those chunks (LLM) │
                └─────────────────────────────────────────────┘
                                   │
                                   ▼
                       grounded answer + sources
```

- **Embedding model:** `all-MiniLM-L6-v2` (small, fast, free).
- **Vector search:** FAISS with cosine similarity.
- **Answer model:** OpenAI `gpt-4o-mini` if you set an API key, otherwise a free
  local `flan-t5-base` model — so it runs with **zero cost** out of the box.

---

## Setup

```bash
# 1. clone and enter the project
git clone https://github.com/Chandana18G/rag-document-chatbot.git
cd rag-document-chatbot

# 2. (recommended) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. install dependencies
pip install -r requirements.txt
```

Optional — to use OpenAI for higher-quality answers:

```bash
cp .env.example .env      # then paste your key into .env
```

---

## Usage

```bash
# 1. add your files to the documents/ folder (a sample is included)

# 2. build the search index (run again whenever documents change)
python ingest.py

# 3a. ask from the command line
python ask.py "What is retrieval-augmented generation?"

# 3b. or launch the web chat UI
streamlit run app.py
```

---

## Project structure

| File | What it does |
|------|--------------|
| `rag.py` | Core engine: read, chunk, embed, retrieve, generate. |
| `ingest.py` | Builds the FAISS vector index from `documents/`. |
| `ask.py` | Command-line question answering. |
| `app.py` | Streamlit web chat interface. |
| `documents/` | Put your source files here. |
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

- Add citation highlighting (show which chunk each sentence came from).
- Swap FAISS for a hosted vector database (e.g. Chroma, Pinecone).
- Deploy the Streamlit app to the cloud.

---

*Built by Chandana Gurusiddappa — M.Sc. Data Science & AI.
[GitHub](https://github.com/Chandana18G) · [LinkedIn](https://linkedin.com/in/chandana-gurusiddappa-785563223)*

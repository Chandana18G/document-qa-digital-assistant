# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

A small Retrieval-Augmented Generation (RAG) app that answers questions about
local documents (PDF / Markdown / TXT). Pipeline: read → chunk → embed
(`all-MiniLM-L6-v2`) → FAISS index → retrieve top-k → generate an answer with
OpenAI `gpt-4o-mini` (if `OPENAI_API_KEY` is set) or local `google/flan-t5-base`.

## Layout

| File | Role |
|------|------|
| `rag.py` | All core logic. Module-level config constants (`DOCS_DIR`, `INDEX_DIR`, `EMBED_MODEL_NAME`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`) and lazily-loaded model singletons. |
| `ingest.py` | Thin entry point: calls `rag.build_index()`. |
| `ask.py` | CLI: one-shot (`python ask.py "question"`) or interactive loop. |
| `app.py` | Streamlit chat UI; calls `rag.answer_question()` per message. |
| `architecture.svg` | Diagram of the pipeline (not currently linked from the README). |

Data directories (not in git, created at runtime):
- `documents/` — source files the user drops in. Read recursively.
- `index/` — `faiss.index` + `chunks.pkl` (chunk texts and source filenames, index-aligned).

## Commands

```bash
pip install -r requirements.txt
mkdir -p documents          # add .pdf/.md/.txt files here
python ingest.py            # (re)build index/ — required after any document change
python ask.py "question"    # CLI
streamlit run app.py        # web UI at http://localhost:8501
```

There are no tests, linters, or CI configured yet.

## Key invariants

- `chunks[i]` and `sources[i]` in `chunks.pkl` must stay aligned with FAISS row `i`.
  Any change to chunking or metadata must write all three together.
- Embeddings are L2-normalised and stored in `IndexFlatIP`, so inner product = cosine
  similarity. Query embeddings must also use `normalize_embeddings=True`.
- The embedding model used at query time must match the one used at ingest time;
  changing `EMBED_MODEL_NAME` requires re-running `ingest.py`.
- `DOCS_DIR` / `INDEX_DIR` are relative paths — scripts must be run from the repo root.
- Heavy imports (`pypdf`, `openai`, `transformers`) are deferred inside functions on
  purpose so importing `rag` stays fast. Keep it that way.

## Known issues / gotchas

- `answer_question()` reloads the FAISS index and pickle from disk on every call
  (including every Streamlit message). Not cached.
- An index built from documents that yield zero text (e.g. scanned PDFs) will crash in
  `build_index` (`embeddings.shape[1]` on an empty array).
- The local flan-t5 path truncates the prompt at 1024 tokens from the right, which would
  drop the question itself if the context is long.
- No similarity threshold: the top-k chunks are always sent to the LLM even when
  irrelevant.
- Sources are keyed by `path.name`, so same-named files in different subfolders collide;
  no page numbers are kept for PDFs.
- `transformers`/`torch` are only pulled in transitively via `sentence-transformers`.
- `index/chunks.pkl` is loaded with `pickle` — never load an index from an untrusted source.

## Conventions

- Plain functions, no classes; module docstrings explain usage at the top of each script.
- Keep the zero-cost local fallback working — the app must run without an API key.
- Don't commit `documents/`, `index/`, `.env`, or model caches.

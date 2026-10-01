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
| `architecture.svg` | Diagram of the pipeline, embedded in the README. |

Data directories (resolved relative to `rag.py`; override with `RAG_DOCS_DIR` / `RAG_INDEX_DIR`):
- `documents/` — source files the user drops in. Read recursively. Only `sample.md` is tracked.
- `index/` — `faiss.index` + `chunks.pkl` (chunk texts and source filenames, index-aligned).

## Commands

```bash
pip install -r requirements.txt
# add .pdf/.md/.txt files to documents/ (sample.md included)
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
- `load_index()` caches the index in memory, keyed on the files' mtimes, so re-ingesting
  is picked up without a restart. Streamlit reuses imported modules across reruns, so this
  cache (and the model singletons) persist in the web app too.
- The local flan-t5 prompt is fitted to `LOCAL_MAX_INPUT_TOKENS` by dropping the
  lowest-ranked contexts (`fit_prompt`) — never truncate the prompt itself, the question
  is at the end.
- `read_documents()` skips unreadable/empty files with a warning on stderr; `build_index()`
  exits with a clear message if nothing usable remains.
- Heavy imports (`pypdf`, `openai`, `transformers`) are deferred inside functions on
  purpose so importing `rag` stays fast. Keep it that way.

## Known issues / gotchas

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

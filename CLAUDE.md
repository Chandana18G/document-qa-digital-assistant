# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

A small Retrieval-Augmented Generation (RAG) app that answers questions about
local documents (PDF / Markdown / TXT). Pipeline: read → structure-aware chunk →
embed (`all-MiniLM-L6-v2`) → FAISS index. At question time: rewrite follow-ups using
chat history → hybrid retrieve (FAISS + BM25, reciprocal rank fusion) → cross-encoder
re-rank → relevance threshold → generate with the first available of Claude
(`ANTHROPIC_API_KEY`), OpenAI (`OPENAI_API_KEY`), Ollama (running, model pulled), or
local `google/flan-t5-large`. `LLM_PROVIDER` forces one.

## Layout

| File | Role |
|------|------|
| `rag.py` | All core logic. Module-level config constants (paths, chunking, `TOP_K`, `CANDIDATES`, re-rank/threshold, provider/model names, `TEMPERATURE` — most overridable by env var) and lazily-loaded model singletons. `answer_question(question, history, stream=..., provider=..., top_k=..., ...)` returns a dict: `answer` (str, or an iterator of text pieces when `stream=True`), `contexts`, `sources`, `search_query`, `provider`. |
| `ingest.py` | Thin entry point: calls `rag.build_index()`. |
| `ask.py` | CLI: one-shot (`python ask.py "question"`) or interactive loop that keeps history for follow-ups. Streams the answer, then lists the numbered passages. |
| `app.py` | Streamlit chat UI. Sidebar: upload files + rebuild index, per-session settings (provider, model, temperature, top-k, re-rank, threshold), clear/export chat. Streams answers; each has a "Show passages" expander numbered like the citations. |
| `.streamlit/config.toml` | Disables Streamlit's file watcher (see Known issues). |
| `architecture.svg` | Diagram of the pipeline, embedded in the README. |
| `tests/` | Unit tests (pytest). |
| `evaluation/` | SQuAD 2.0 evaluation: `squad.py` (corpus + questions), `run_eval.py`, `figures.py`. |
| `results/metrics.json`, `figures/` | Output of the last evaluation run, plus the app screenshot `figures/app.png`. |

Data directories (resolved relative to `rag.py`; override with `RAG_DOCS_DIR` / `RAG_INDEX_DIR`):
- `documents/` — source files the user drops in. Read recursively. Only `sample.md` is tracked.
- `index/` — `faiss.index` + `chunks.pkl` (`chunks`, `sources`, `pages` lists, index-aligned).
  `sources` are paths relative to the docs dir (`hr/policy.pdf`); `pages` are 1-based PDF
  page numbers or `None`. PDFs are chunked page by page so every chunk has one page.
  The BM25 keyword index is not stored; `load_index()` rebuilds it from the chunk texts.
  `load_index()` returns a dict (`index`, `chunks`, `sources`, `pages`, `bm25`) and
  defaults `pages` for indexes built before it existed.

## Commands

```bash
pip install -r requirements.txt
# add .pdf/.md/.txt files to documents/ (sample.md included)
python ingest.py            # (re)build index/ — required after any document change
python ask.py "question"    # CLI
streamlit run app.py        # web UI at http://localhost:8501
python -m pytest            # unit tests (~1 min, no model downloads)
python -m evaluation.run_eval [--quick]   # SQuAD 2.0 evaluation -> results/, figures/
python -m evaluation.figures              # redraw the charts from results/metrics.json
```

- `tests/test_rag.py` replaces the embedder with a bag-of-words stand-in (`FakeEmbedder`),
  so tests exercise the real indexing and search code without downloading models. No
  linters or CI yet.
- `evaluation/` downloads SQuAD 2.0 dev into `evaluation/data/` and builds indexes in
  `evaluation/work/` (both gitignored). It calls `rag.retrieve()` / `rag.answer_question()`
  directly, so it measures the real pipeline. Re-run it after changing retrieval, models
  or thresholds, and update the README's results section from `results/metrics.json`.

## Key invariants

- `chunks[i]`, `sources[i]` and `pages[i]` in `chunks.pkl` must stay aligned with FAISS
  row `i`. Any change to chunking or metadata must write them all together.
- Embeddings are L2-normalised and stored in `IndexFlatIP`, so inner product = cosine
  similarity. Query embeddings must also use `normalize_embeddings=True`.
- The embedding model used at query time must match the one used at ingest time;
  changing `EMBED_MODEL_NAME` requires re-running `ingest.py`.
- `load_index()` caches the index in memory, keyed on the files' mtimes, so re-ingesting
  is picked up without a restart. Streamlit reuses imported modules across reruns, so this
  cache (and the model singletons) persist in the web app too.
- Chunks start with their Markdown section path (`Title > Section`) and never cross a
  section boundary; that prefix is part of the embedded and BM25-indexed text.
- `_stream_local()` runs `generate()` on one reused worker thread (`_local_worker`), never a
  new thread per answer — that leaked ~80 MB of PyTorch per-thread memory each time. Its
  `run()` always calls `streamer.end()` and the caller calls `job.result()`, so a failing
  `generate()` raises instead of hanging (regression-tested).
- `retrieve()` sorts by cross-encoder score only when re-ranking. Without re-ranking it
  keeps the fused (RRF) order — re-sorting by cosine would drop the BM25 half of the hybrid
  search (regression-tested in `test_hybrid_order_is_kept_without_reranking`).
- Relevance thresholds are calibrated per scoring mode: `MIN_RERANK_SCORE` applies to
  cross-encoder logits (relevant ≳ -2, off-topic ≈ -10 on `sample.md`), `MIN_SIMILARITY`
  to cosine similarity when `RAG_RERANK=0`. If nothing passes, `answer_question` returns
  `NO_ANSWER` without calling the LLM. Re-check both after changing either model.
- Follow-ups: hosted/Ollama providers rewrite the question via `condense_question`.
  flan-t5 can't rewrite reliably, so for `local` the app retrieves with both the bare
  follow-up and "previous question + follow-up" and merges the results.
- The local flan-t5 prompt is fitted to `LOCAL_MAX_INPUT_TOKENS` by dropping the
  lowest-ranked contexts (`fit_prompt`, which returns the kept contexts) — never truncate
  the prompt itself, the question is at the end. The returned `contexts` are exactly the
  passages in the prompt, in prompt order, so `[n]` citations map to `contexts[n-1]`.
- Hosted/Ollama prompts ask for `[n]` citations; the flan-t5 prompt doesn't (it can't
  follow the instruction).
- Every provider is a `_stream_*` generator; `complete()` joins one. Per-request settings
  are passed as arguments, never by mutating module globals — Streamlit sessions share
  the module.
- `read_documents()` skips unreadable/empty files with a warning on stderr; `build_index()`
  exits with a clear message if nothing usable remains.
- Heavy or optional imports (`pypdf`, `openai`, `anthropic`, `transformers` model
  classes) are deferred inside functions; `anthropic`/`openai` are optional installs.
  Keep it that way.
- The Claude call uses `client.beta.messages.create` with server-side refusal
  fallbacks (`fallbacks="default"`, beta `server-side-fallback-2026-07-01`) and
  `effort: "low"`; Opus 5.5 rejects `temperature` and disabled thinking, so don't add them.
- Ollama is called over its REST API with `urllib` — no extra dependency.

## Known issues / gotchas

- flan-t5 (the zero-setup fallback) gives terse answers and often answers only the
  first part of a multi-part question; Ollama or a hosted model fixes this.
- Streamlit's file watcher is off (`.streamlit/config.toml`): scanning `transformers`'
  lazy modules imported thousands of submodules, flooded the log with `torchvision`
  errors and could stall the server mid-answer until the browser reconnected, losing the
  chat. Restart the server after editing `app.py`.
- Uploads in the web UI are saved to `documents/` by file name only — an upload with the
  same name as an existing top-level file replaces it.
- `transformers`/`torch` are only pulled in transitively via `sentence-transformers`.
  Importing `rag` still takes several seconds because `sentence_transformers` is imported
  at module level.
- `index/chunks.pkl` is loaded with `pickle` — never load an index from an untrusted source.

## Conventions

- Plain functions, no classes; module docstrings explain usage at the top of each script.
- Keep the zero-cost local fallback working — the app must run without an API key.
- Don't commit `documents/`, `index/`, `.env`, or model caches.

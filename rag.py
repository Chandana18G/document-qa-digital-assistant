"""
rag.py - the core Retrieval-Augmented Generation (RAG) engine.

The whole RAG idea in four steps:
    1. INGEST   : read your documents and split them into small chunks.
    2. EMBED    : turn each chunk into a vector and store it in a FAISS index.
    3. RETRIEVE : embed the question and find the closest chunks.
    4. GENERATE : ask a language model to answer using ONLY those chunks.
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Paths default to folders next to this file, so the scripts work from any
# working directory. Override with RAG_DOCS_DIR / RAG_INDEX_DIR.
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", BASE_DIR / "documents"))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", BASE_DIR / "index"))
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 4
# flan-t5 was trained on 512-token inputs; longer prompts get truncated.
LOCAL_MAX_INPUT_TOKENS = 512

_embedder = None
_local_tok = None
_local_model = None
_index_cache = {"key": None, "value": None}


def get_embedder():
    """Load the embedding model lazily so importing this file stays fast."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


# STEP 1 - read and chunk documents
def read_documents(docs_dir: Path = DOCS_DIR):
    """Read every .txt, .md and .pdf file in docs_dir into raw text.

    Files that can't be read (corrupt or encrypted PDFs, etc.) are skipped
    with a warning instead of aborting the whole ingest.
    """
    texts = []
    for path in sorted(docs_dir.glob("**/*")):
        suffix = path.suffix.lower()
        if suffix not in {".txt", ".md", ".pdf"} or not path.is_file():
            continue
        try:
            if suffix == ".pdf":
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                content = "\n".join((page.extract_text() or "") for page in reader.pages)
            else:
                content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"WARNING: skipping '{path}': {e}", file=sys.stderr)
            continue
        if not content.strip():
            print(f"WARNING: no text extracted from '{path}' (scanned PDF?)", file=sys.stderr)
            continue
        texts.append((path.name, content))
    return texts


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split one document into overlapping windows of characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


# STEP 2 - embed chunks and build the FAISS index
def build_index(docs_dir: Path = DOCS_DIR, index_dir: Path = INDEX_DIR):
    """Read docs -> chunk -> embed -> save a searchable FAISS index to disk."""
    docs = read_documents(docs_dir)
    if not docs:
        raise SystemExit(f"No readable documents found in '{docs_dir}'. Add files first.")

    chunks, sources = [], []
    for name, text in docs:
        for c in chunk_text(text):
            chunks.append(c)
            sources.append(name)
    if not chunks:
        raise SystemExit("Documents were read but produced no text chunks.")

    print(f"Read {len(docs)} document(s) -> {len(chunks)} chunk(s). Embedding...")
    embeddings = get_embedder().encode(chunks, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    with open(index_dir / "chunks.pkl", "wb") as f:
        pickle.dump({"chunks": chunks, "sources": sources}, f)
    print(f"Saved index with {len(chunks)} chunks to '{index_dir}'.")


# STEP 3 - retrieve the most relevant chunks for a question
def load_index(index_dir: Path = INDEX_DIR):
    """Load the index, reusing the in-memory copy until the files on disk change."""
    index_path, store_path = index_dir / "faiss.index", index_dir / "chunks.pkl"
    key = (str(index_path.resolve()), index_path.stat().st_mtime_ns, store_path.stat().st_mtime_ns)
    if _index_cache["key"] != key:
        index = faiss.read_index(str(index_path))
        with open(store_path, "rb") as f:
            store = pickle.load(f)
        _index_cache["value"] = (index, store["chunks"], store["sources"])
        _index_cache["key"] = key
    return _index_cache["value"]


def retrieve(question: str, index, chunks, sources, top_k: int = TOP_K):
    q_vec = get_embedder().encode([question], normalize_embeddings=True)
    q_vec = np.asarray(q_vec, dtype="float32")
    scores, ids = index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append({"text": chunks[idx], "source": sources[idx], "score": float(score)})
    return results


# STEP 4 - generate an answer grounded in the retrieved chunks
def build_prompt(question: str, contexts: list) -> str:
    context_block = "\n\n".join(
        f"[{i+1}] (from {c['source']})\n{c['text']}" for i, c in enumerate(contexts)
    )
    return (
        "You are a helpful assistant. Answer the question using ONLY the context "
        "below. If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def fit_prompt(question: str, contexts: list, count_tokens, max_tokens: int) -> str:
    """Build a prompt within max_tokens by dropping the lowest-ranked contexts.

    Truncating the finished prompt would cut off its end - the question itself -
    so trim the context instead.
    """
    contexts = list(contexts)
    while contexts:
        prompt = build_prompt(question, contexts)
        if count_tokens(prompt) <= max_tokens:
            return prompt
        contexts.pop()
    return build_prompt(question, [])


def generate_answer(question: str, contexts: list) -> str:
    """Use OpenAI if an API key is set, otherwise a free local model."""
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": build_prompt(question, contexts)}],
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()

    # Free local fallback. Load the model directly (not via transformers
    # "pipeline") so it works across all transformers versions.
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    global _local_tok, _local_model
    if _local_model is None:
        _local_tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
        _local_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    prompt = fit_prompt(
        question, contexts,
        count_tokens=lambda p: len(_local_tok(p).input_ids),
        max_tokens=LOCAL_MAX_INPUT_TOKENS,
    )
    inputs = _local_tok(prompt, return_tensors="pt", truncation=True, max_length=LOCAL_MAX_INPUT_TOKENS)
    output_ids = _local_model.generate(**inputs, max_new_tokens=256)
    return _local_tok.decode(output_ids[0], skip_special_tokens=True).strip()


def answer_question(question: str, index_dir: Path = INDEX_DIR):
    """Full pipeline: retrieve then generate. Returns (answer, sources)."""
    index, chunks, sources = load_index(index_dir)
    contexts = retrieve(question, index, chunks, sources)
    answer = generate_answer(question, contexts)
    used = sorted({c["source"] for c in contexts})
    return answer, used

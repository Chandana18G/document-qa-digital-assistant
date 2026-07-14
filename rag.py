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
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path("documents")
INDEX_DIR = Path("index")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 4

_embedder = None
_local_tok = None
_local_model = None


def get_embedder():
    """Load the embedding model lazily so importing this file stays fast."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


# STEP 1 - read and chunk documents
def read_documents(docs_dir: Path = DOCS_DIR):
    """Read every .txt, .md and .pdf file in docs_dir into raw text."""
    texts = []
    for path in sorted(docs_dir.glob("**/*")):
        if path.suffix.lower() in {".txt", ".md"}:
            texts.append((path.name, path.read_text(encoding="utf-8", errors="ignore")))
        elif path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            content = "\n".join((page.extract_text() or "") for page in reader.pages)
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
        raise SystemExit(f"No documents found in '{docs_dir}'. Add files first.")

    chunks, sources = [], []
    for name, text in docs:
        for c in chunk_text(text):
            chunks.append(c)
            sources.append(name)

    print(f"Read {len(docs)} document(s) -> {len(chunks)} chunk(s). Embedding...")
    embeddings = get_embedder().encode(chunks, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_dir.mkdir(exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    with open(index_dir / "chunks.pkl", "wb") as f:
        pickle.dump({"chunks": chunks, "sources": sources}, f)
    print(f"Saved index with {len(chunks)} chunks to '{index_dir}'.")


# STEP 3 - retrieve the most relevant chunks for a question
def load_index(index_dir: Path = INDEX_DIR):
    index = faiss.read_index(str(index_dir / "faiss.index"))
    with open(index_dir / "chunks.pkl", "rb") as f:
        store = pickle.load(f)
    return index, store["chunks"], store["sources"]


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


def generate_answer(question: str, contexts: list) -> str:
    """Use OpenAI if an API key is set, otherwise a free local model."""
    prompt = build_prompt(question, contexts)

    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
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
    inputs = _local_tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    output_ids = _local_model.generate(**inputs, max_new_tokens=256)
    return _local_tok.decode(output_ids[0], skip_special_tokens=True).strip()


def answer_question(question: str, index_dir: Path = INDEX_DIR):
    """Full pipeline: retrieve then generate. Returns (answer, sources)."""
    index, chunks, sources = load_index(index_dir)
    contexts = retrieve(question, index, chunks, sources)
    answer = generate_answer(question, contexts)
    used = sorted({c["source"] for c in contexts})
    return answer, used

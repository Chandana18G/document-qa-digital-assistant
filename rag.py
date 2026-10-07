"""
rag.py - the core Retrieval-Augmented Generation (RAG) engine.

The whole RAG idea in four steps:
    1. INGEST   : read your documents and split them into small chunks.
    2. EMBED    : turn each chunk into a vector and store it in a FAISS index.
    3. RETRIEVE : find the closest chunks - vector search + keyword (BM25) search,
                  re-ranked by a cross-encoder and filtered by a relevance threshold.
    4. GENERATE : ask a language model to answer using ONLY those chunks.

Follow-up questions ("and what about equipment?") are first rewritten into
standalone questions using the chat history, so retrieval sees the full intent.
"""

import json
import math
import os
import pickle
import re
import sys
import urllib.request
from collections import Counter, defaultdict
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
CHUNK_SIZE = 800      # max characters per chunk (excluding the section-title prefix)
CHUNK_OVERLAP = 150   # trailing sentences/paragraphs of up to this many chars repeat in the next chunk
TOP_K = 4             # chunks sent to the LLM
CANDIDATES = 20       # chunks each search (vector, BM25) proposes before re-ranking

# Re-ranking and relevance threshold. With re-ranking on, MIN_RERANK_SCORE applies to the
# cross-encoder score (a logit: > 0 relevant, around -10 clearly irrelevant); with it off,
# MIN_SIMILARITY applies to the cosine similarity of the query and chunk embeddings.
RERANK = os.getenv("RAG_RERANK", "1") != "0"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MIN_RERANK_SCORE = float(os.getenv("RAG_MIN_RERANK_SCORE", "-4.0"))
MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.35"))
NO_ANSWER = "I couldn't find anything about that in your documents."

# Generation. LLM_PROVIDER = auto | anthropic | openai | ollama | local.
# "auto" picks the first available: ANTHROPIC_API_KEY -> OPENAI_API_KEY -> a running
# Ollama with OLLAMA_MODEL pulled -> the local flan-t5 model (free, no setup).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
LOCAL_MODEL_NAME = os.getenv("RAG_LOCAL_MODEL", "google/flan-t5-large")
# flan-t5 was trained on 512-token inputs; longer prompts get truncated.
LOCAL_MAX_INPUT_TOKENS = 512
TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.1"))   # used by OpenAI and Ollama
HISTORY_TURNS = 3     # previous Q&A pairs used to rewrite a follow-up question

_embedder = None
_reranker = None
_local_models = {}   # model name -> (tokenizer, model)
_index_cache = {"key": None, "value": None}


def get_embedder():
    """Load the embedding model lazily so importing this file stays fast."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL_NAME)
    return _reranker


# STEP 1 - read and chunk documents
DOC_TYPES = {".txt", ".md", ".pdf"}


def read_documents(docs_dir: Path = DOCS_DIR):
    """Read every .txt, .md and .pdf file in docs_dir.

    Returns (source, page, text) tuples. source is the path relative to docs_dir
    (e.g. "hr/policy.pdf"), so same-named files in different folders stay distinct.
    PDFs yield one tuple per page with a 1-based page number; other files one
    tuple with page None. Files that can't be read (corrupt or encrypted PDFs,
    etc.) are skipped with a warning instead of aborting the whole ingest.
    """
    texts = []
    for path in sorted(docs_dir.glob("**/*")):
        suffix = path.suffix.lower()
        if suffix not in DOC_TYPES or not path.is_file():
            continue
        source = path.relative_to(docs_dir).as_posix()
        try:
            if suffix == ".pdf":
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                parts = [(source, n, page.extract_text() or "") for n, page in enumerate(reader.pages, 1)]
            else:
                parts = [(source, None, path.read_text(encoding="utf-8", errors="ignore"))]
        except Exception as e:
            print(f"WARNING: skipping '{path}': {e}", file=sys.stderr)
            continue
        parts = [p for p in parts if p[2].strip()]
        if not parts:
            print(f"WARNING: no text extracted from '{path}' (scanned PDF?)", file=sys.stderr)
            continue
        texts.extend(parts)
    return texts


def source_label(source: str, page) -> str:
    """How a chunk's origin is shown to users, e.g. "hr/policy.pdf p. 4"."""
    return f"{source} p. {page}" if page else source


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sections(text: str):
    """Split Markdown into (heading path, body) sections, e.g. ("Policy > Equipment", "...").

    Text without Markdown headings (plain text, PDFs) comes back as one untitled section.
    """
    sections, path, body = [], [], []
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            sections.append((" > ".join(t for _, t in path), "\n".join(body)))
            level = len(m.group(1))
            path = [(lvl, t) for lvl, t in path if lvl < level] + [(level, m.group(2))]
            body = []
        else:
            body.append(line)
    sections.append((" > ".join(t for _, t in path), "\n".join(body)))
    return [(title, b.strip()) for title, b in sections if b.strip()]


def _units(body: str, size: int):
    """Break a section into pieces no longer than size: paragraphs, then sentences,
    then (for a single huge sentence) plain character windows."""
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            yield para
            continue
        for sent in _SENTENCE_RE.split(para):
            for i in range(0, len(sent), size):
                piece = sent[i:i + size].strip()
                if piece:
                    yield piece


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split one document into chunks that follow its structure.

    Chunks never cross a section boundary and break between paragraphs or
    sentences, not mid-word. Each chunk starts with its section title so the
    embedding and the LLM both know where the passage comes from. Consecutive
    chunks of one section share up to `overlap` characters of whole sentences.
    """
    chunks = []
    for title, body in split_sections(text):
        prefix = f"{title}\n" if title else ""
        current = []
        for unit in _units(body, size):
            if current and len("\n".join(current + [unit])) > size:
                chunks.append(prefix + "\n".join(current))
                # Repeat the tail of the finished chunk, as long as it stays within
                # the overlap budget and still leaves room for the new unit.
                carry = []
                for prev in reversed(current):
                    candidate = [prev] + carry
                    if (len("\n".join(candidate)) > overlap
                            or len("\n".join(candidate + [unit])) > size):
                        break
                    carry = candidate
                current = carry
            current.append(unit)
        if current:
            chunks.append(prefix + "\n".join(current))
    return chunks


# STEP 2 - embed chunks and build the FAISS index
def build_index(docs_dir: Path = DOCS_DIR, index_dir: Path = INDEX_DIR):
    """Read docs -> chunk -> embed -> save a searchable FAISS index to disk.

    Returns (number of documents, number of chunks).
    """
    docs = read_documents(docs_dir)
    if not docs:
        raise SystemExit(f"No readable documents found in '{docs_dir}'. Add files first.")

    # chunks[i], sources[i] and pages[i] describe FAISS row i - keep them aligned.
    chunks, sources, pages = [], [], []
    for source, page, text in docs:
        for c in chunk_text(text):
            chunks.append(c)
            sources.append(source)
            pages.append(page)
    if not chunks:
        raise SystemExit("Documents were read but produced no text chunks.")

    n_docs = len({s for s, _, _ in docs})
    print(f"Read {n_docs} document(s) -> {len(chunks)} chunk(s). Embedding...")
    embeddings = get_embedder().encode(chunks, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    with open(index_dir / "chunks.pkl", "wb") as f:
        pickle.dump({"chunks": chunks, "sources": sources, "pages": pages}, f)
    print(f"Saved index with {len(chunks)} chunks to '{index_dir}'.")
    return n_docs, len(chunks)


# STEP 3 - retrieve the most relevant chunks for a question
def _tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def build_bm25(docs):
    """Build an Okapi BM25 keyword index - it catches exact terms, names and
    numbers that embedding search can miss."""
    postings = defaultdict(list)   # term -> [(doc id, term frequency)]
    lengths = []
    for i, doc in enumerate(docs):
        tokens = _tokenize(doc)
        lengths.append(len(tokens))
        for term, tf in Counter(tokens).items():
            postings[term].append((i, tf))
    n = len(docs)
    idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in postings.items()}
    return {"postings": postings, "idf": idf, "lengths": lengths,
            "avg_len": (sum(lengths) / n) or 1.0}


def bm25_top(bm25, query: str, k: int, k1: float = 1.5, b: float = 0.75):
    """Ids of the k chunks with the highest BM25 score for query, best first."""
    scores = defaultdict(float)
    for term in set(_tokenize(query)):
        for i, tf in bm25["postings"].get(term, ()):
            norm = tf + k1 * (1 - b + b * bm25["lengths"][i] / bm25["avg_len"])
            scores[i] += bm25["idf"][term] * tf * (k1 + 1) / norm
    return sorted(scores, key=scores.get, reverse=True)[:k]


def index_exists(index_dir: Path = INDEX_DIR) -> bool:
    return (index_dir / "faiss.index").exists() and (index_dir / "chunks.pkl").exists()


def load_index(index_dir: Path = INDEX_DIR):
    """Load the index, reusing the in-memory copy until the files on disk change.

    Returns a dict with the FAISS "index", the aligned "chunks", "sources" and
    "pages" lists, and a "bm25" keyword index rebuilt from the chunk texts
    (it is not stored on disk).
    """
    index_path, store_path = index_dir / "faiss.index", index_dir / "chunks.pkl"
    key = (str(index_path.resolve()), index_path.stat().st_mtime_ns, store_path.stat().st_mtime_ns)
    if _index_cache["key"] != key:
        with open(store_path, "rb") as f:
            store = pickle.load(f)
        store["index"] = faiss.read_index(str(index_path))
        # Indexes built before page numbers were stored have no "pages" list.
        store.setdefault("pages", [None] * len(store["chunks"]))
        store["bm25"] = build_bm25(store["chunks"])
        _index_cache["value"] = store
        _index_cache["key"] = key
    return _index_cache["value"]


def _relevance(result) -> float:
    """The score retrieve() ranked a result by: cross-encoder if re-ranking, else cosine."""
    return result["rerank_score"] if result["rerank_score"] is not None else result["similarity"]


def retrieve(question: str, store, top_k: int = TOP_K, rerank: bool = RERANK, min_score: float = None):
    """Hybrid search: vector + BM25 candidates merged with reciprocal rank fusion,
    re-ranked by a cross-encoder, then filtered by the relevance threshold.

    min_score defaults to MIN_RERANK_SCORE with re-ranking, MIN_SIMILARITY without.
    Returns up to top_k dicts (text, source, page, label, similarity, rerank_score),
    best first - possibly none, if nothing in the documents is relevant enough.
    """
    index, chunks = store["index"], store["chunks"]
    q_vec = np.asarray(get_embedder().encode([question], normalize_embeddings=True), dtype="float32")
    _, ids = index.search(q_vec, min(CANDIDATES, len(chunks)))
    rankings = [[int(i) for i in ids[0] if i != -1], bm25_top(store["bm25"], question, CANDIDATES)]

    fused = defaultdict(float)
    for ranking in rankings:
        for rank, i in enumerate(ranking):
            fused[i] += 1.0 / (60 + rank)
    candidates = sorted(fused, key=fused.get, reverse=True)

    results = [{
        "text": chunks[i],
        "source": store["sources"][i],
        "page": store["pages"][i],
        "label": source_label(store["sources"][i], store["pages"][i]),
        "similarity": float(np.dot(q_vec[0], index.reconstruct(i))),
        "rerank_score": None,
    } for i in candidates]

    if rerank and results:
        scores = get_reranker().predict([(question, r["text"]) for r in results])
        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)
    if min_score is None:
        min_score = MIN_RERANK_SCORE if rerank else MIN_SIMILARITY
    results.sort(key=_relevance, reverse=True)
    return [r for r in results if _relevance(r) >= min_score][:top_k]


# STEP 4 - generate an answer grounded in the retrieved chunks
def build_prompt(question: str, contexts: list, cite: bool = True) -> str:
    context_block = "\n\n".join(
        f"[{i+1}] (from {c['label']})\n{c['text']}" for i, c in enumerate(contexts)
    )
    citation_rule = (
        "After each sentence, cite the numbers of the passages it is based on in "
        "square brackets, like [1] or [2][3]. "
    ) if cite else ""
    return (
        "You are a helpful assistant. Answer the question using ONLY the context "
        "below. Answer every part of the question. " + citation_rule +
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def fit_prompt(question: str, contexts: list, count_tokens, max_tokens: int, cite: bool = True):
    """Build a prompt within max_tokens by dropping the lowest-ranked contexts.

    Truncating the finished prompt would cut off its end - the question itself -
    so trim the context instead. Returns (prompt, contexts kept).
    """
    contexts = list(contexts)
    while contexts:
        prompt = build_prompt(question, contexts, cite)
        if count_tokens(prompt) <= max_tokens:
            return prompt, contexts
        contexts.pop()
    return build_prompt(question, [], cite), []


PROVIDERS = ["anthropic", "openai", "ollama", "local"]
DEFAULT_MODELS = {
    "anthropic": ANTHROPIC_MODEL,
    "openai": OPENAI_MODEL,
    "ollama": OLLAMA_MODEL,
    "local": LOCAL_MODEL_NAME,
}


def _ollama_ready(model: str = OLLAMA_MODEL) -> bool:
    """True if an Ollama server is reachable and has the model pulled."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=0.5) as resp:
            names = {m["name"] for m in json.load(resp).get("models", [])}
    except Exception:
        return False
    return model in names or f"{model}:latest" in names


def get_provider(preferred: str = None) -> str:
    """Resolve "auto" (or None, meaning LLM_PROVIDER) to a concrete provider."""
    preferred = (preferred or LLM_PROVIDER).lower()
    if preferred != "auto":
        if preferred not in PROVIDERS:
            raise ValueError(f"Unknown LLM provider '{preferred}'. Use one of: auto, {', '.join(PROVIDERS)}.")
        return preferred
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if _ollama_ready():
        return "ollama"
    return "local"


# Each _stream_* function yields the answer text piece by piece as it is generated.
def _stream_anthropic(prompt: str, model: str, temperature: float):
    import anthropic
    client = anthropic.Anthropic()
    # temperature is not sent: current Claude models don't accept it.
    # "fallbacks" re-runs the request on another model if this one declines
    # for safety reasons; on a stream the answer just continues.
    with client.beta.messages.stream(
        model=model,
        max_tokens=16000,
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        yield from stream.text_stream
        final = stream.get_final_message()
    if final.stop_reason == "refusal":
        yield "\n\n(The model declined to answer this question.)"


def _stream_openai(prompt: str, model: str, temperature: float):
    from openai import OpenAI
    stream = OpenAI().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def _stream_ollama(prompt: str, model: str, temperature: float):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {"temperature": temperature},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA_HOST}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=300)
    except OSError as e:
        raise RuntimeError(
            f"Could not get an answer from Ollama at {OLLAMA_HOST} ({e}). Is `ollama serve` "
            f"running and `{model}` pulled (`ollama pull {model}`)?"
        ) from e
    with resp:
        for line in resp:   # one JSON object per line
            if line.strip():
                yield json.loads(line).get("message", {}).get("content", "")


def _load_local(name: str = LOCAL_MODEL_NAME):
    # Load the model directly (not via transformers "pipeline") so it works
    # across all transformers versions.
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    if name not in _local_models:
        _local_models[name] = (AutoTokenizer.from_pretrained(name),
                               AutoModelForSeq2SeqLM.from_pretrained(name))
    return _local_models[name]


def _stream_local(prompt: str, model: str, temperature: float):
    # Greedy decoding - temperature is not used by the local model.
    from threading import Thread
    from transformers import TextIteratorStreamer
    tok, lm = _load_local(model)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=LOCAL_MAX_INPUT_TOKENS)
    streamer = TextIteratorStreamer(tok, skip_special_tokens=True)
    worker = Thread(target=lm.generate, kwargs=dict(**inputs, max_new_tokens=256, streamer=streamer),
                    daemon=True)
    worker.start()
    yield from streamer
    worker.join()   # don't let the generator finish while torch is still running


_STREAMERS = {
    "anthropic": _stream_anthropic,
    "openai": _stream_openai,
    "ollama": _stream_ollama,
    "local": _stream_local,
}


def stream_completion(prompt: str, provider: str = None, model: str = None, temperature: float = TEMPERATURE):
    provider = get_provider(provider)
    return _STREAMERS[provider](prompt, model or DEFAULT_MODELS[provider], temperature)


def complete(prompt: str, provider: str = None, model: str = None, temperature: float = TEMPERATURE) -> str:
    return "".join(stream_completion(prompt, provider, model, temperature)).strip()


def make_prompt(question: str, contexts: list, provider: str, model: str = None):
    """Build the answer prompt for provider. Returns (prompt, contexts actually included).

    flan-t5 gets a prompt trimmed to its input limit and no citation instruction
    (it can't follow one); the other models get every context and cite them.
    """
    if provider == "local":
        tok, _ = _load_local(model or LOCAL_MODEL_NAME)
        return fit_prompt(question, contexts, count_tokens=lambda p: len(tok(p).input_ids),
                          max_tokens=LOCAL_MAX_INPUT_TOKENS, cite=False)
    return build_prompt(question, contexts), contexts


def condense_question(question: str, history: list, provider: str = None, model: str = None,
                      temperature: float = TEMPERATURE) -> str:
    """Rewrite a follow-up question into a standalone one using the chat history.

    history is a list of (question, answer) pairs, oldest first. Returns the
    question unchanged when there is no history or the rewrite comes back empty.
    """
    if not history:
        return question
    convo = "\n".join(
        f"User: {q}\nAssistant: {a[:300]}" for q, a in history[-HISTORY_TURNS:]
    )
    prompt = (
        "Rewrite the follow-up question so it can be understood without the "
        "conversation, replacing pronouns and references with what they refer to. "
        "If it is already standalone, return it unchanged. Output only the question.\n\n"
        f"Conversation:\n{convo}\n\n"
        f"Follow-up question: {question}\n\n"
        "Standalone question:"
    )
    rewritten = complete(prompt, provider, model, temperature).strip().strip('"').splitlines()
    return rewritten[0].strip() if rewritten and rewritten[0].strip() else question


def answer_question(question: str, history: list = None, *, stream: bool = False,
                    provider: str = None, model: str = None, temperature: float = TEMPERATURE,
                    top_k: int = TOP_K, rerank: bool = RERANK, min_score: float = None,
                    index_dir: Path = INDEX_DIR):
    """Full pipeline: rewrite follow-ups, retrieve, then generate.

    history is a list of previous (question, answer) pairs, oldest first. The
    keyword arguments override the module defaults for this call only.

    Returns a dict:
        answer        the answer text - or, with stream=True, an iterator of text pieces
        contexts      the passages given to the model, numbered [1], [2], ... in the prompt
        sources       their labels in that order, without duplicates
        search_query  what was searched for (the rewritten follow-up, if any)
        provider      the provider that answered
    """
    provider = get_provider(provider)
    store = load_index(index_dir)
    search = dict(top_k=top_k, rerank=rerank, min_score=min_score)
    if history and provider == "local":
        # flan-t5 can't reliably rewrite questions. Search with the follow-up alone
        # first (it may be self-contained); only if nothing relevant turns up, add
        # the previous question for context (the follow-up may say "it"/"they").
        # The model answers the follow-up itself either way.
        search_query = final_question = question
        contexts = retrieve(question, store, **search)
        if not contexts:
            search_query = f"{history[-1][0]} {question}"
            contexts = retrieve(search_query, store, **search)
    else:
        search_query = final_question = condense_question(question, history, provider, model, temperature)
        contexts = retrieve(search_query, store, **search)

    result = {"contexts": [], "sources": [], "search_query": search_query, "provider": provider}
    if not contexts:
        result["answer"] = iter([NO_ANSWER]) if stream else NO_ANSWER
        return result
    prompt, contexts = make_prompt(final_question, contexts, provider, model)
    pieces = stream_completion(prompt, provider, model, temperature)
    result["answer"] = pieces if stream else "".join(pieces).strip()
    result["contexts"] = contexts
    result["sources"] = list(dict.fromkeys(c["label"] for c in contexts))
    return result

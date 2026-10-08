"""Unit tests for rag.py. No models are downloaded: the embedder is replaced by a
small bag-of-words stand-in, so these run in a few seconds.

    python -m pytest
"""

import hashlib
import os
import re
import time

import numpy as np
import pytest

import rag


class FakeEmbedder:
    """Hashes words into a fixed-size vector: texts sharing words are similar."""
    dim = 256

    def encode(self, texts, normalize_embeddings=True, **_):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


@pytest.fixture
def fake_embedder(monkeypatch):
    monkeypatch.setattr(rag, "get_embedder", lambda: FakeEmbedder())
    monkeypatch.setitem(rag._index_cache, "key", None)


POLICY = """# Remote Work Policy

## Eligibility

Employees may work remotely up to 3 days per week after probation.

## Equipment

The company provides a laptop and a home office allowance of 300 EUR.
"""


@pytest.fixture
def indexed(tmp_path, fake_embedder):
    docs, index = tmp_path / "docs", tmp_path / "index"
    (docs / "hr").mkdir(parents=True)
    (docs / "hr" / "policy.md").write_text(POLICY, encoding="utf-8")
    (docs / "notes.txt").write_text("Penguins live in the southern hemisphere.", encoding="utf-8")
    rag.build_index(docs, index)
    return index


# Chunking
def test_split_sections_tracks_heading_path():
    sections = rag.split_sections("# Doc\nintro\n## A\nalpha\n### A1\ndeep\n## B\nbeta")
    assert sections == [("Doc", "intro"), ("Doc > A", "alpha"), ("Doc > A > A1", "deep"), ("Doc > B", "beta")]


def test_text_without_headings_is_one_untitled_section():
    assert rag.split_sections("just text\n\nmore text") == [("", "just text\n\nmore text")]


def test_chunks_start_with_section_path_and_never_cross_sections():
    chunks = rag.chunk_text(POLICY)
    assert chunks[0].startswith("Remote Work Policy > Eligibility\n")
    assert chunks[1].startswith("Remote Work Policy > Equipment\n")
    assert not any("probation" in c and "laptop" in c for c in chunks)


def test_chunks_respect_size_and_carry_overlap():
    sentences = [f"Sentence number {i} is here." for i in range(40)]
    chunks = rag.chunk_text(" ".join(sentences), size=200, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # The last sentence of one chunk opens the next.
    for a, b in zip(chunks, chunks[1:]):
        assert b.split("\n")[0] in a


def test_a_single_huge_sentence_is_split_into_windows():
    chunks = rag.chunk_text("x" * 450, size=200, overlap=0)
    assert [len(c) for c in chunks] == [200, 200, 50]


# Reading documents
def test_read_documents_uses_relative_paths_and_skips_unusable_files(tmp_path, capsys):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.md").write_text("hello", encoding="utf-8")
    (tmp_path / "b.txt").write_text("world", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    docs = rag.read_documents(tmp_path)
    assert docs == [("b.txt", None, "world"), ("sub/a.md", None, "hello")]
    assert "empty.txt" in capsys.readouterr().err


def test_source_label_adds_pdf_page():
    assert rag.source_label("hr/policy.pdf", 4) == "hr/policy.pdf p. 4"
    assert rag.source_label("notes.md", None) == "notes.md"


# Keyword search
def test_bm25_ranks_the_chunk_with_the_exact_term_first():
    docs = ["the cat sat on the mat", "a laptop allowance of 300 EUR", "dogs bark at night"]
    bm25 = rag.build_bm25(docs)
    assert rag.bm25_top(bm25, "allowance in EUR", 3)[0] == 1
    assert rag.bm25_top(bm25, "zebra", 3) == []


# Index and retrieval
def test_index_keeps_chunks_sources_and_vectors_aligned(indexed):
    store = rag.load_index(indexed)
    assert store["index"].ntotal == len(store["chunks"]) == len(store["sources"]) == len(store["pages"])
    assert set(store["sources"]) == {"hr/policy.md", "notes.txt"}


def test_retrieve_finds_the_relevant_section(indexed):
    store = rag.load_index(indexed)
    results = rag.retrieve("How much is the home office allowance?", store, rerank=False, min_score=0.0)
    assert "300 EUR" in results[0]["text"]
    assert results[0]["label"] == "hr/policy.md"


def test_threshold_filters_off_topic_questions(indexed):
    store = rag.load_index(indexed)
    assert rag.retrieve("quantum chromodynamics lattice", store, rerank=False, min_score=0.3) == []


# Chosen so the vector and BM25 rankings disagree on the middle results.
SOLAR = {
    "a.txt": "solar panels roof energy",
    "b.txt": "solar solar panels panels roof roof energy energy cost cost savings savings",
    "c.txt": "the warranty for inverters lasts ten years",
    "d.txt": "inverter warranty",
    "e.txt": "kitchen ovens bake bread",
}


def test_hybrid_order_is_kept_without_reranking(tmp_path, fake_embedder):
    """Regression test: without re-ranking, results follow the fused (RRF) order of
    the vector and BM25 rankings - not a re-sort by cosine similarity, which used to
    throw the BM25 half away."""
    docs, index = tmp_path / "docs", tmp_path / "index"
    docs.mkdir()
    for name, text in SOLAR.items():
        (docs / name).write_text(text, encoding="utf-8")
    rag.build_index(docs, index)
    store = rag.load_index(index)
    query = "inverter warranty years solar panels"

    q_vec = FakeEmbedder().encode([query]).astype("float32")
    _, ids = store["index"].search(q_vec, len(store["chunks"]))
    fused = {}
    for ranking in ([int(i) for i in ids[0]], rag.bm25_top(store["bm25"], query, rag.CANDIDATES)):
        for rank, i in enumerate(ranking):
            fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank)
    expected = [store["chunks"][i] for i in sorted(fused, key=fused.get, reverse=True)]

    results = rag.retrieve(query, store, rerank=False, min_score=-1.0, top_k=10)
    by_cosine = [r["text"] for r in sorted(results, key=lambda r: r["similarity"], reverse=True)]
    assert [r["text"] for r in results] == expected
    assert by_cosine != expected, "test data must make the two orders differ"


def test_load_index_reloads_after_rebuild(tmp_path, fake_embedder):
    docs, index = tmp_path / "docs", tmp_path / "index"
    docs.mkdir()
    (docs / "a.txt").write_text("first version", encoding="utf-8")
    rag.build_index(docs, index)
    assert rag.load_index(index)["chunks"] == ["first version"]
    (docs / "a.txt").write_text("second version", encoding="utf-8")
    rag.build_index(docs, index)
    later = time.time_ns() + 10**9   # coarse filesystem clocks: make sure the mtimes change
    for name in ("faiss.index", "chunks.pkl"):
        os.utime(index / name, ns=(later, later))
    assert rag.load_index(index)["chunks"] == ["second version"]


def test_build_index_rejects_an_empty_folder(tmp_path, fake_embedder):
    with pytest.raises(SystemExit):
        rag.build_index(tmp_path, tmp_path / "index")


# Prompts
def _ctx(n):
    return [{"label": f"doc{i}.md", "text": f"passage {i} " * 20} for i in range(1, n + 1)]


def test_prompt_numbers_passages_and_restricts_to_context():
    prompt = rag.build_prompt("What?", _ctx(2))
    assert "[1] (from doc1.md)" in prompt and "[2] (from doc2.md)" in prompt
    assert "using ONLY the context" in prompt and "say you don't know" in prompt
    assert prompt.rstrip().endswith("Question: What?\n\nAnswer:")
    assert "cite" in prompt and "cite" not in rag.build_prompt("What?", _ctx(2), cite=False)


def test_fit_prompt_drops_lowest_ranked_contexts_and_keeps_the_question():
    count_words = lambda p: len(p.split())
    full = count_words(rag.build_prompt("Where?", _ctx(3)))
    prompt, kept = rag.fit_prompt("Where?", _ctx(3), count_words, max_tokens=full - 10)
    assert [c["label"] for c in kept] == ["doc1.md", "doc2.md"]
    assert prompt.endswith("Question: Where?\n\nAnswer:")
    _, none = rag.fit_prompt("Where?", _ctx(3), count_words, max_tokens=5)
    assert none == []


# Providers and the full pipeline
def test_get_provider(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(rag, "_ollama_ready", lambda *a: False)
    assert rag.get_provider("auto") == "local"
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    assert rag.get_provider("auto") == "openai"
    assert rag.get_provider("ollama") == "ollama"
    with pytest.raises(ValueError):
        rag.get_provider("gpt-9")


def test_no_relevant_passage_means_no_llm_call(indexed, monkeypatch):
    def fail(*a, **k):
        raise AssertionError("the LLM must not be called")
    monkeypatch.setattr(rag, "stream_completion", fail)
    result = rag.answer_question("quantum chromodynamics lattice", provider="openai",
                                 rerank=False, min_score=0.3, index_dir=indexed)
    assert result["answer"] == rag.NO_ANSWER and result["contexts"] == []


def test_answer_uses_the_passages_it_returns(indexed, monkeypatch):
    seen = {}

    def fake_stream(prompt, model, temperature):
        seen["prompt"] = prompt
        yield "300 EUR [1]"
    monkeypatch.setitem(rag._STREAMERS, "openai", fake_stream)
    result = rag.answer_question("home office allowance", provider="openai", rerank=False,
                                 min_score=0.0, top_k=2, index_dir=indexed)
    assert result["answer"] == "300 EUR [1]"
    assert result["contexts"][0]["text"] in seen["prompt"]
    assert result["sources"] == list(dict.fromkeys(c["label"] for c in result["contexts"]))


def test_local_model_errors_are_raised_not_hung(monkeypatch):
    """Regression test: if generate() fails on the worker thread (e.g. out of memory),
    the caller gets the error instead of waiting forever for streamed text."""
    class Tok:
        def __call__(self, prompt, **_):
            return {"input_ids": [[1, 2, 3]]}

        def decode(self, ids, **_):
            return ""

    class LM:
        def generate(self, **_):
            raise RuntimeError("not enough memory")

    monkeypatch.setattr(rag, "_load_local", lambda name=None: (Tok(), LM()))
    outcome = {}

    def consume():
        try:
            "".join(rag._stream_local("prompt", "fake-model", 0.0))
        except Exception as e:
            outcome["error"] = e

    # A daemon thread with a time limit, so a regression fails the test instead of hanging it.
    import threading
    reader = threading.Thread(target=consume, daemon=True)
    reader.start()
    reader.join(timeout=30)
    assert not reader.is_alive(), "streaming hung after generate() failed"
    assert "not enough memory" in str(outcome.get("error"))


def test_condense_question_without_history_returns_it_unchanged():
    assert rag.condense_question("And equipment?", []) == "And equipment?"

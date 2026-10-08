"""
run_eval.py - measure the RAG pipeline on SQuAD 2.0.

Usage (from the repository root):
    python -m evaluation.run_eval                 # full run, about an hour on a laptop CPU
    python -m evaluation.run_eval --quick         # retrieval, refusals, chunk size (~40 min)
    python -m evaluation.run_eval --answers-only  # add the answer evaluation to saved results

What it measures:
    1. Retrieval   how often each search method puts the passage that holds the
                   answer in the top 1 / top 4 (TOP_K, what the LLM sees).
    2. Refusals    whether the relevance threshold answers questions about the
                   documents and refuses questions about topics that aren't there.
    3. Chunk size  the retrieval / context-length trade-off for 400, 800, 1200 chars.
    4. Answers     exact match and token F1 of the free local model (flan-t5-large),
                   and whether it says "I don't know" to unanswerable questions.

Writes results/metrics.json and the charts in figures/ (via evaluation.figures).
"""

import argparse
import json
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

import rag
from evaluation import squad

ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = Path(__file__).resolve().parent / "work"
RESULTS = ROOT / "results" / "metrics.json"
CHUNK_SIZES = {400: 75, 800: rag.CHUNK_OVERLAP, 1200: 225}   # chunk size -> overlap
DEPTH = 2 * rag.CANDIDATES   # how far down each ranking is scored (MRR)
THRESHOLDS = [round(t, 2) for t in np.arange(-12, 8.01, 0.5)]


# Corpus and indexes
def build_indexes(indexed_articles):
    docs_dir = WORK_DIR / "docs"
    squad.write_corpus(indexed_articles, docs_dir)
    stats = {}
    for size, overlap in CHUNK_SIZES.items():
        index_dir = WORK_DIR / f"index-{size}"
        if not rag.index_exists(index_dir):   # the corpus is deterministic, so reuse old indexes
            rag.build_index(docs_dir, index_dir, chunk_size=size, overlap=overlap)
        store = rag.load_index(index_dir)
        chunks = store["chunks"]
        stats[size] = {"documents": len(set(store["sources"])), "chunks": len(chunks),
                       "mean_chunk_chars": round(float(np.mean([len(c) for c in chunks])), 1)}
    return stats


def gold_texts(store, items):
    """Attach to each answerable question the chunk texts that contain its answer sentence.

    Questions whose answer sentence no chunk contains whole (a sentence longer than
    the chunk size) are dropped and counted.
    """
    normalized = [squad.normalize_space(c) for c in store["chunks"]]
    kept, dropped = [], 0
    for q in items:
        gold = {store["chunks"][i] for i, c in enumerate(normalized) if q["sentence"] in c}
        if gold:
            kept.append({**q, "gold": gold})
        else:
            dropped += 1
    return kept, dropped


# 1. Retrieval
def rank_metrics(rankings, items):
    """hit@1, hit@TOP_K and MRR over DEPTH, from rankings of chunk texts (best first)."""
    hits1, hitsk, rr = [], [], []
    for ranking, q in zip(rankings, items):
        pos = next((i for i, text in enumerate(ranking[:DEPTH]) if text in q["gold"]), None)
        hits1.append(pos == 0)
        hitsk.append(pos is not None and pos < rag.TOP_K)
        rr.append(0.0 if pos is None else 1.0 / (pos + 1))
    return {"hit@1": round(float(np.mean(hits1)), 4), f"hit@{rag.TOP_K}": round(float(np.mean(hitsk)), 4),
            "mrr": round(float(np.mean(rr)), 4), "n": len(items)}


def vector_rankings(store, texts):
    vecs = np.asarray(rag.get_embedder().encode(texts, normalize_embeddings=True, batch_size=64), dtype="float32")
    sims, ids = store["index"].search(vecs, rag.CANDIDATES)
    rankings = [[store["chunks"][i] for i in row if i != -1] for row in ids]
    return rankings, sims[:, 0]


def bm25_rankings(store, texts):
    return [[store["chunks"][i] for i in rag.bm25_top(store["bm25"], t, rag.CANDIDATES)] for t in texts]


def pipeline_results(store, texts):
    """What rag.retrieve() returns for each question - threshold off, every candidate
    kept - without and with re-ranking: (hybrid, reranked) lists of result dicts.

    Same steps as rag.retrieve(), but batched: one encode call for all questions and
    large cross-encoder batches. Calling retrieve() 1,000 times took hours on a laptop.
    check_matches_retrieve() confirms the two agree.
    """
    chunks, index = store["chunks"], store["index"]
    vectors = index.reconstruct_n(0, index.ntotal)
    q_vecs = np.asarray(rag.get_embedder().encode(texts, normalize_embeddings=True, batch_size=64), dtype="float32")
    _, ids = index.search(q_vecs, min(rag.CANDIDATES, len(chunks)))
    hybrid = []
    for q, row, text in zip(q_vecs, ids, texts):
        fused = {}
        for ranking in ([int(i) for i in row if i != -1], rag.bm25_top(store["bm25"], text, rag.CANDIDATES)):
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank)
        hybrid.append([{"text": chunks[i], "label": rag.source_label(store["sources"][i], store["pages"][i]),
                        "similarity": float(np.dot(q, vectors[i])), "rerank_score": None}
                       for i in sorted(fused, key=fused.get, reverse=True)])

    pairs = [(t, r["text"]) for t, res in zip(texts, hybrid) for r in res]
    scores = iter(rag.get_reranker().predict(pairs, batch_size=64, show_progress_bar=False))
    reranked = []
    for res in hybrid:
        scored = [{**r, "rerank_score": float(next(scores))} for r in res]
        reranked.append(sorted(scored, key=lambda r: r["rerank_score"], reverse=True))
    return hybrid, reranked


def check_matches_retrieve(store, texts, hybrid, reranked, n=10):
    """Compare the batched results with rag.retrieve() on n questions; return its latency."""
    t0 = time.time()
    same = 0
    for text, h, r in zip(texts[:n], hybrid, reranked):
        real_h = rag.retrieve(text, store, top_k=rag.TOP_K, rerank=False, min_score=-float("inf"))
        real_r = rag.retrieve(text, store, top_k=rag.TOP_K, rerank=True, min_score=-float("inf"))
        same += ([x["text"] for x in real_h] == [x["text"] for x in h[:rag.TOP_K]]
                 and [x["text"] for x in real_r] == [x["text"] for x in r[:rag.TOP_K]])
    print(f"  batched results match rag.retrieve() top {rag.TOP_K} on {same}/{min(n, len(texts))} questions")
    return same, (time.time() - t0) / (2 * min(n, len(texts)))


def evaluate_retrieval(index_dir, items, label):
    store = rag.load_index(index_dir)
    items, dropped = gold_texts(store, items)
    texts = [q["question"] for q in items]
    print(f"[{label}] scoring {len(items)} questions ({dropped} dropped)...")
    hybrid, reranked = pipeline_results(store, texts)
    matches, seconds = check_matches_retrieve(store, texts, hybrid, reranked)
    # Before the fix, retrieve() re-sorted the fused candidates by cosine similarity when
    # re-ranking was off, which undid the BM25 half of the hybrid ranking.
    before_fix = [[r["text"] for r in sorted(res, key=lambda r: r["similarity"], reverse=True)] for res in hybrid]
    out = {
        "hybrid_before_fix": rank_metrics(before_fix, items),
        "hybrid": rank_metrics([[r["text"] for r in res] for res in hybrid], items),
        "hybrid_rerank": rank_metrics([[r["text"] for r in res] for res in reranked], items),
        "dropped_no_gold_chunk": dropped,
        "seconds_per_query": round(seconds, 3),
        "batched_matches_retrieve": f"{matches}/{min(10, len(texts))}",
    }
    return out, items, store, reranked


# 2. Refusals
def top_scores(results):
    """Best re-rank score and best cosine similarity per question (-inf if no results)."""
    rerank = [max((r["rerank_score"] for r in res), default=-float("inf")) for res in results]
    cosine = [max((r["similarity"] for r in res), default=-float("inf")) for res in results]
    return np.array(rerank), np.array(cosine)


def refusal_curve(in_scores, in_found, out_scores, thresholds):
    """For each threshold: share of in-corpus questions answered (and answered with the
    right passage among those shown), and share of out-of-corpus questions answered."""
    rows = []
    for t in thresholds:
        answered = in_scores >= t
        rows.append({"threshold": t,
                     "in_corpus_answered": round(float(answered.mean()), 4),
                     "in_corpus_answered_with_gold": round(float((answered & in_found).mean()), 4),
                     "out_of_corpus_answered": round(float((out_scores >= t).mean()), 4)})
    return rows


def at_threshold(curve, t):
    return min(curve, key=lambda row: abs(row["threshold"] - t))


# 3. Contexts that fit flan-t5's input
def contexts_that_fit(results, items):
    # Only the tokenizer is needed - loading the 3 GB model here can exhaust a laptop's RAM.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(rag.LOCAL_MODEL_NAME)
    kept = []
    for res, q in zip(results, items):
        top = res[:rag.TOP_K]
        _, ctx = rag.fit_prompt(q["question"], top, lambda p: len(tok(p).input_ids),
                                rag.LOCAL_MAX_INPUT_TOKENS, cite=False)
        kept.append(len(ctx))
    return round(float(np.mean(kept)), 2)


# 4. Answers (SQuAD answer normalisation)
def _norm(text: str) -> str:
    text = "".join(ch for ch in text.lower() if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(pred: str, golds) -> float:
    return float(any(_norm(pred) == _norm(g) for g in golds))


def token_f1(pred: str, golds) -> float:
    best = 0.0
    p = _norm(pred).split()
    for g in golds:
        g = _norm(g).split()
        common = sum((Counter(p) & Counter(g)).values())
        if common:
            precision, recall = common / len(p), common / len(g)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


_DONT_KNOW = re.compile(r"don'?t know|do not know|unanswerable|not (?:mentioned|stated|provided|"
                        r"given|in the (?:context|passage|text))|no answer|cannot be answered|"
                        r"couldn'?t find", re.I)


def is_refusal(answer: str) -> bool:
    return answer == rag.NO_ANSWER or bool(_DONT_KNOW.search(answer))


def evaluate_answers(index_dir, answerable, unanswerable):
    # One JSON line per answer, so an interrupted run (or a laptop going to sleep)
    # resumes where it stopped instead of starting over.
    log_path = WORK_DIR / f"answers-{rag.LOCAL_MODEL_NAME.replace('/', '_')}.jsonl"
    done = {}
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["id"]] = row
    rows = {"answerable": [], "unanswerable": []}
    with open(log_path, "a", encoding="utf-8") as log:
        for kind, items in (("answerable", answerable), ("unanswerable", unanswerable)):
            for n, q in enumerate(items, 1):
                if q["id"] not in done:
                    t0 = time.time()
                    res = rag.answer_question(q["question"], provider="local", index_dir=index_dir)
                    done[q["id"]] = {"id": q["id"], "answer": res["answer"], "seconds": time.time() - t0,
                                     "golds": q["answers"]}
                    log.write(json.dumps(done[q["id"]]) + "\n")
                    log.flush()
                rows[kind].append(done[q["id"]])
                if n % 10 == 0:
                    print(f"  {kind}: {n}/{len(items)}")
    ans = rows["answerable"]
    return {
        "model": rag.LOCAL_MODEL_NAME,
        "answerable": {
            "n": len(ans),
            "exact_match": round(float(np.mean([exact_match(r["answer"], r["golds"]) for r in ans])), 4),
            "f1": round(float(np.mean([token_f1(r["answer"], r["golds"]) for r in ans])), 4),
            "refused": round(float(np.mean([is_refusal(r["answer"]) for r in ans])), 4),
        },
        "unanswerable": {
            "n": len(rows["unanswerable"]),
            "refused": round(float(np.mean([is_refusal(r["answer"]) for r in rows["unanswerable"]])), 4),
        },
        "seconds_per_answer": round(float(np.mean([r["seconds"] for r in ans + rows["unanswerable"]])), 2),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--quick", action="store_true", help="skip the answer evaluation")
    parser.add_argument("--answers-only", action="store_true",
                        help="only run the answer evaluation and add it to the saved results")
    parser.add_argument("--questions", type=int, default=1000, help="in-corpus questions for retrieval")
    parser.add_argument("--chunk-questions", type=int, default=300, help="questions per chunk size")
    parser.add_argument("--answer-questions", type=int, default=150, help="answerable questions to answer")
    parser.add_argument("--unanswerable", type=int, default=100, help="unanswerable questions to answer")
    parser.add_argument("--resume", action="store_true", help="reuse stages a stopped run finished")
    args = parser.parse_args(argv)

    articles = squad.load_squad()
    indexed, held_out = squad.split_articles(articles)
    in_corpus = squad.sample(squad.questions(indexed, answerable=True), args.questions)
    out_of_corpus = squad.sample(squad.questions(held_out, answerable=True)
                                 + squad.questions(held_out, answerable=False), args.questions // 2)
    main_dir = WORK_DIR / f"index-{rag.CHUNK_SIZE}"

    if args.answers_only:
        # A separate step: the answer model needs ~3 GB of RAM on its own.
        results = json.loads(RESULTS.read_text(encoding="utf-8"))
        if not rag.index_exists(main_dir):
            squad.write_corpus(indexed, WORK_DIR / "docs")
            rag.build_index(WORK_DIR / "docs", main_dir)
        print("Generating answers with the local model (this is the slow part)...")
        unanswerable = squad.sample(squad.questions(indexed, answerable=False), args.unanswerable)
        results["answers"] = evaluate_answers(main_dir, in_corpus[:args.answer_questions], unanswerable)
        save(results)
        return

    corpus = build_indexes(indexed)

    # --resume: reuse the stages a stopped run already finished (same question sample).
    partial = {}
    if args.resume and (WORK_DIR / "partial.json").exists():
        partial = json.loads((WORK_DIR / "partial.json").read_text(encoding="utf-8"))
        if partial.get("refusals", {}).get("n_out_of_corpus") != len(out_of_corpus):
            partial = {}
    if partial:
        print("Resuming: reusing retrieval and refusal results from evaluation/work/partial.json")
        retrieval, refusals = partial["retrieval"], partial["refusals"]
    else:
        retrieval, refusals = retrieval_and_refusals(main_dir, in_corpus, out_of_corpus)
        checkpoint({"retrieval": retrieval, "refusals": refusals})

    # 3. Chunk size: the same questions against each index.
    chunk_sizes = {int(k): v for k, v in partial.get("chunk_sizes", {}).items()}
    subset = in_corpus[:args.chunk_questions]
    for size in CHUNK_SIZES:
        if size in chunk_sizes and chunk_sizes[size]["n"] >= len(subset) - 10:
            continue
        res, size_items, _, size_results = evaluate_retrieval(WORK_DIR / f"index-{size}", subset, f"{size} chars")
        chunk_sizes[size] = {**corpus[size], **res["hybrid_rerank"],
                             "contexts_fitting_flan_t5": contexts_that_fit(size_results, size_items)}
        checkpoint({"retrieval": retrieval, "refusals": refusals, "chunk_sizes": chunk_sizes})
    chunk_sizes = {size: chunk_sizes[size] for size in CHUNK_SIZES}

    results = {
        "dataset": "SQuAD 2.0 dev (Rajpurkar et al., 2018), CC BY-SA 4.0",
        "corpus": {"indexed_articles": [squad.title(a) for a in indexed],
                   "held_out_articles": [squad.title(a) for a in held_out]},
        "settings": {"embed_model": rag.EMBED_MODEL_NAME, "rerank_model": rag.RERANK_MODEL_NAME,
                     "top_k": rag.TOP_K, "candidates": rag.CANDIDATES, "chunk_size": rag.CHUNK_SIZE},
        "retrieval": retrieval,
        "refusals": refusals,
        "chunk_sizes": chunk_sizes,
    }
    if RESULTS.exists() and "answers" in (old := json.loads(RESULTS.read_text(encoding="utf-8"))):
        results["answers"] = old["answers"]   # keep an earlier answer evaluation
    save(results)   # retrieval results are kept even if the answer step fails

    # 4. Answers, in a fresh process so the retrieval models' memory is released first.
    if not args.quick:
        import subprocess
        subprocess.run([sys.executable, "-m", "evaluation.run_eval", "--answers-only",
                        "--questions", str(args.questions),
                        "--answer-questions", str(args.answer_questions),
                        "--unanswerable", str(args.unanswerable)], check=True, cwd=ROOT)


def retrieval_and_refusals(main_dir, in_corpus, out_of_corpus):
    # 1. Retrieval: the four search methods at the default chunk size.
    retrieval, items, store, reranked = evaluate_retrieval(main_dir, in_corpus, f"{rag.CHUNK_SIZE} chars")
    texts = [q["question"] for q in items]
    vec_rank, _ = vector_rankings(store, texts)
    retrieval["vector"] = rank_metrics(vec_rank, items)
    retrieval["bm25"] = rank_metrics(bm25_rankings(store, texts), items)

    # 2. Refusals: top scores for in-corpus vs out-of-corpus questions.
    in_rerank, in_cos = top_scores(reranked)
    in_found = np.array([any(r["text"] in q["gold"] for r in res[:rag.TOP_K]) for res, q in zip(reranked, items)])
    print(f"[refusals] scoring {len(out_of_corpus)} questions about held-out articles...")
    _, out_results = pipeline_results(store, [q["question"] for q in out_of_corpus])
    out_rerank, _ = top_scores(out_results)
    _, out_cos = vector_rankings(store, [q["question"] for q in out_of_corpus])
    rerank_curve = refusal_curve(in_rerank, in_found, out_rerank, THRESHOLDS)
    cos_curve = refusal_curve(in_cos, in_found, out_cos, [round(t, 2) for t in np.arange(0, 1.001, 0.05)])
    refusals = {
        "n_in_corpus": len(items), "n_out_of_corpus": len(out_of_corpus),
        "rerank_curve": rerank_curve, "cosine_curve": cos_curve,
        "default_rerank": at_threshold(rerank_curve, rag.MIN_RERANK_SCORE),
        "default_cosine": at_threshold(cos_curve, rag.MIN_SIMILARITY),
    }

    return retrieval, refusals


def checkpoint(partial):
    """Keep finished stages on disk while the run continues."""
    (WORK_DIR / "partial.json").write_text(json.dumps(partial, indent=2, default=str), encoding="utf-8")


def save(results):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {RESULTS}")
    from evaluation import figures
    figures.make_all(json.loads(RESULTS.read_text(encoding="utf-8")))


if __name__ == "__main__":
    sys.exit(main())

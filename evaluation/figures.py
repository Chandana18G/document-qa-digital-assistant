"""
figures.py - charts for results/metrics.json, written to figures/.

Usage:
    python -m evaluation.figures      # redraw from the saved metrics
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"

# Dark theme matching the portfolio; colours checked for contrast and colour-blind separation.
BG, GRID, AXIS, TEXT, MUTED = "#110e1a", "#2a2240", "#4a4066", "#ece8f5", "#a59dbb"
VIOLET, TEAL, MAGENTA = "#8b5cf6", "#0f9f8f", "#d946ef"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": AXIS, "axes.labelcolor": MUTED, "axes.titlecolor": TEXT,
        "axes.titlesize": 17, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 34, "axes.labelsize": 13, "axes.grid": True, "grid.color": GRID,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 12.5, "ytick.labelsize": 12,
        "xtick.major.size": 0, "ytick.major.size": 0, "text.color": TEXT,
        "legend.frameon": False, "legend.fontsize": 12.5, "font.size": 12.5,
    })


def subtitle(ax, text: str):
    ax.text(0, 1.025, text, transform=ax.transAxes, color=MUTED, fontsize=12.5, va="bottom")


def save(fig, name: str):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"Wrote figures/{name}.png")


def retrieval_chart(m):
    r = m["retrieval"]
    k = m["settings"]["top_k"]
    methods = [("bm25", "BM25\n(keywords)"), ("vector", "Vector\n(embeddings)"),
               ("hybrid", "Hybrid\n(BM25 + vector, RRF)"), ("hybrid_rerank", "Hybrid +\ncross-encoder re-rank")]
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    width = 0.36
    for j, (metric, colour, label) in enumerate([("hit@1", VIOLET, "Right passage ranked 1st"),
                                                  (f"hit@{k}", TEAL, f"Right passage in the top {k} (what the LLM sees)")]):
        xs = [i + (j - 0.5) * width for i in range(len(methods))]
        vals = [r[key][metric] * 100 for key, _ in methods]
        ax.bar(xs, vals, width, color=colour, label=label, edgecolor=BG, linewidth=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.2, f"{v:.0f}%", ha="center", fontsize=13, color=TEXT)
    ax.set_xticks(range(len(methods)), [name for _, name in methods])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Questions (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.set_title("Finding the passage that holds the answer")
    subtitle(ax, f"{r['hybrid_rerank']['n']} SQuAD 2.0 questions over 25 Wikipedia articles "
                 f"({m['chunk_sizes'][str(m['settings']['chunk_size'])]['chunks']} chunks)")
    fig.tight_layout()
    save(fig, "retrieval")


def refusal_chart(m):
    f = m["refusals"]
    curve = f["rerank_curve"]
    xs = [row["threshold"] for row in curve]
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    series = [("in_corpus_answered", TEAL, "Questions about the documents: answered"),
              ("out_of_corpus_answered", MAGENTA, "Questions about other topics: answered (should be 0)")]
    for key, colour, label in series:
        ax.plot(xs, [row[key] * 100 for row in curve], color=colour, lw=3, label=label)
    d = f["default_rerank"]
    ax.axvline(d["threshold"], color=MUTED, ls="--", lw=1.2)
    ax.text(d["threshold"] + 0.2, 52, f"app default ({d['threshold']:g})", color=MUTED, fontsize=12)
    for key, colour, _ in series:
        v = d[key] * 100
        ax.scatter([d["threshold"]], [v], s=70, color=colour, edgecolor=BG, linewidth=2, zorder=3)
        ax.text(d["threshold"] - 0.3, v + 2.5, f"{v:.0f}%", color=TEXT, fontsize=13, ha="right")
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(0, 105)
    ax.set_xlabel("Relevance threshold (cross-encoder score of the best passage)")
    ax.set_ylabel("Questions answered (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2)
    ax.set_title("Answering what it can, refusing what it can't")
    subtitle(ax, f"{f['n_in_corpus']} questions about the indexed articles · "
                 f"{f['n_out_of_corpus']} about 10 held-out articles · below the threshold the app says it couldn't find anything")
    fig.tight_layout()
    save(fig, "refusals")


def chunk_size_chart(m):
    c = m["chunk_sizes"]
    k = m["settings"]["top_k"]
    sizes = sorted(c, key=int)
    labels = [f"{s} chars\n({c[s]['chunks']} chunks)" for s in sizes]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2))
    panels = [(axes[0], [c[s][f"hit@{k}"] * 100 for s in sizes], VIOLET, "{:.0f}%",
               f"Right passage in the top {k}", "Questions (%)", 105),
              (axes[1], [c[s]["contexts_fitting_flan_t5"] for s in sizes], TEAL, "{:.1f}",
               f"Passages that fit flan-t5's 512 tokens (of {k})", "Passages per question", k + 0.5)]
    for ax, vals, colour, fmt, title, ylabel, top in panels:
        ax.bar(range(len(sizes)), vals, 0.55, color=colour, edgecolor=BG, linewidth=2)
        for i, v in enumerate(vals):
            ax.text(i, v + top * 0.015, fmt.format(v), ha="center", fontsize=13, color=TEXT)
        ax.set_xticks(range(len(sizes)), labels)
        ax.set_ylim(0, top)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=15, pad=14)
    fig.suptitle("Chunk size barely changes retrieval, but bigger chunks crowd out the local model",
                 x=0.012, ha="left", fontsize=17, fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=4)
    save(fig, "chunk_size")


def make_all(metrics=None):
    if metrics is None:
        metrics = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
    # JSON keys are strings; normalise chunk sizes so lookups work after a reload too.
    metrics["chunk_sizes"] = {str(k): v for k, v in metrics["chunk_sizes"].items()}
    apply_style()
    retrieval_chart(metrics)
    refusal_chart(metrics)
    chunk_size_chart(metrics)


if __name__ == "__main__":
    make_all()

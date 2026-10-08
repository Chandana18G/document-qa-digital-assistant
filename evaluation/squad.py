"""
squad.py - turn the SQuAD 2.0 dev set into an evaluation corpus for the RAG pipeline.

SQuAD 2.0 (Rajpurkar et al., 2018; CC BY-SA 4.0) has Wikipedia paragraphs with
crowd-written questions. Some questions are answerable from their paragraph; the
rest are written to look answerable but are not.

The 35 dev articles are split at random: 25 become the document collection (one
Markdown file per article) and 10 are held out, so their questions are about
topics that are NOT in the documents - the assistant should refuse those.
"""

import json
import random
import re
import urllib.request
from pathlib import Path

import rag

URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
DATA_DIR = Path(__file__).resolve().parent / "data"


def load_squad(path: Path = DATA_DIR / "dev-v2.0.json"):
    """The SQuAD 2.0 dev set as a list of articles, downloaded on first use (~4 MB)."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading SQuAD 2.0 dev set to {path} ...")
        urllib.request.urlretrieve(URL, path)
    return json.loads(path.read_text(encoding="utf-8"))["data"]


def split_articles(articles, n_indexed: int = 25, seed: int = 42):
    """(indexed, held_out) articles, chosen at random but reproducibly."""
    shuffled = list(articles)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:n_indexed], shuffled[n_indexed:]


def title(article) -> str:
    return article["title"].replace("_", " ")


def write_corpus(articles, docs_dir: Path):
    """Write one Markdown file per article: a title heading, then its paragraphs."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    for old in docs_dir.glob("*.md"):
        old.unlink()
    for a in articles:
        body = "\n\n".join(p["context"].strip() for p in a["paragraphs"])
        name = re.sub(r"[^\w-]+", "_", a["title"]) + ".md"
        (docs_dir / name).write_text(f"# {title(a)}\n\n{body}\n", encoding="utf-8")


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def answer_sentence(context: str, start: int) -> str:
    """The sentence of context that contains character offset start.

    Uses the same sentence boundaries as the chunker, so a chunk either contains
    the whole sentence or none of it.
    """
    begin = 0
    for m in rag._SENTENCE_RE.finditer(context):
        if m.start() >= start:
            return context[begin:m.start()]
        begin = m.end()
    return context[begin:]


def questions(articles, answerable: bool):
    """Question dicts for the given articles.

    Each has: id, question, title, answers (list of accepted strings) and, for
    answerable questions, sentence - the sentence that contains the first answer,
    used to decide which chunks count as correct retrievals.
    """
    out = []
    for a in articles:
        for p in a["paragraphs"]:
            for q in p["qas"]:
                if q["is_impossible"] == answerable:
                    continue
                item = {"id": q["id"], "question": q["question"].strip(), "title": title(a),
                        "answers": [x["text"] for x in q["answers"]]}
                if answerable:
                    first = q["answers"][0]
                    item["sentence"] = normalize_space(answer_sentence(p["context"], first["answer_start"]))
                out.append(item)
    return out


def sample(items, n: int, seed: int = 42):
    """n items at random (all of them if there are fewer), in a reproducible order."""
    items = list(items)
    random.Random(seed).shuffle(items)
    return items[:n]

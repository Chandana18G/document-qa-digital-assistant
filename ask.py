"""
ask.py — ask questions from the command line.

Usage:
    python ask.py "What is this document about?"
    python ask.py            # then type questions interactively; blank line quits

Answers are printed as they are generated. In interactive mode follow-up
questions ("and what about equipment?") use the earlier questions and answers
as context.
"""

import sys
from rag import answer_question


def ask_once(question: str, history: list = None) -> str:
    result = answer_question(question, history, stream=True)
    if result["search_query"] != question:
        print(f"\n(Searched for: {result['search_query']})")
    print("\nAnswer:")
    pieces = []
    for piece in result["answer"]:
        pieces.append(piece)
        print(piece, end="", flush=True)
    print("\n\nPassages:" if result["contexts"] else "\n\nPassages: none")
    for i, c in enumerate(result["contexts"], 1):
        print(f"  [{i}] {c['label']}")
    print()
    return "".join(pieces).strip()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ask_once(" ".join(sys.argv[1:]))
    else:
        print("Ask a question (press Enter on an empty line to quit).")
        history = []
        while True:
            q = input("\n> ").strip()
            if not q:
                break
            history.append((q, ask_once(q, history)))

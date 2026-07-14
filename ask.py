"""
ask.py — ask questions from the command line.

Usage:
    python ask.py "What is this document about?"
    python ask.py            # then type questions interactively; blank line quits
"""

import sys
from rag import answer_question


def ask_once(question: str):
    answer, sources = answer_question(question)
    print("\nAnswer:\n" + answer)
    print("\nSources: " + ", ".join(sources) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ask_once(" ".join(sys.argv[1:]))
    else:
        print("Ask a question (press Enter on an empty line to quit).")
        while True:
            q = input("\n> ").strip()
            if not q:
                break
            ask_once(q)

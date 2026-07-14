"""
ingest.py — run this ONCE (and again whenever you change your documents).

It reads everything in the documents/ folder, splits it into chunks, turns
each chunk into a vector, and saves a searchable index into the index/ folder.

Usage:
    python ingest.py
"""

from rag import build_index

if __name__ == "__main__":
    build_index()

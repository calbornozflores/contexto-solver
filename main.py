"""Main entry point for the contexto-solver.

Demonstrates loading pre-computed embeddings and running similarity queries.
"""

from __future__ import annotations

import sys

from src.similarity import most_similar
from src.storage import embeddings_exist, load_embeddings
from src.utils import timer


def main() -> None:
    """Run an interactive similarity query loop."""
    if not embeddings_exist():
        print(
            "Embeddings not found. Please run the generation script first:\n"
            "  python scripts/generate_embeddings.py"
        )
        sys.exit(1)

    print("Loading embeddings …")
    with timer("Load"):
        embeddings, words = load_embeddings()
    print(f"  Loaded {len(words):,} words, embedding dim={embeddings.shape[1]}.")

    print("\nContexto Solver — semantic similarity search")
    print("Type a word to find the most similar words (Ctrl-C to quit).\n")

    try:
        while True:
            query = input("Query: ").strip()
            if not query:
                continue
            results = most_similar(query, top_k=10, embeddings=embeddings, words=words)
            print(f"\nTop-10 most similar to '{query}':")
            for rank, (word, score) in enumerate(results, start=1):
                print(f"  {rank:>2}. {word:<20} {score:.4f}")
            print()
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    main()

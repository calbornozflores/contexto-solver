"""CLI script to generate and persist word embeddings.

Usage::

    python scripts/generate_embeddings.py [--model MODEL] [--batch-size N]

The script skips generation if the embedding files already exist unless
``--force`` is passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so that ``src`` is importable when the
# script is run from any working directory.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.embedding_model import generate_embeddings  # noqa: E402
from src.loader import load_words  # noqa: E402
from src.storage import embeddings_exist, save_embeddings  # noqa: E402
from src.utils import timer  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and store word embeddings for the contexto-solver."
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Number of words per embedding batch (default: 512)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-generate embeddings even if files already exist",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the embedding generation script."""
    args = parse_args()

    if not args.force and embeddings_exist():
        print("Embedding files already exist. Use --force to regenerate.")
        return

    print("Loading words …")
    words_set = load_words()
    words = sorted(words_set)
    print(f"  Loaded {len(words):,} words.")

    with timer("Embedding generation"):
        embeddings = generate_embeddings(words, model_name=args.model, batch_size=args.batch_size)

    print(f"  Embeddings shape: {embeddings.shape}")

    print("Saving embeddings …")
    save_embeddings(embeddings, words)
    print("  Done. Files saved to data/embeddings.npy and data/words.txt")


if __name__ == "__main__":
    main()

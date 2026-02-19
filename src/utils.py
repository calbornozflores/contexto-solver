"""Utility helpers for the contexto-solver project."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


@contextmanager
def timer(label: str = "Elapsed") -> Generator[None, None, None]:
    """Context manager that prints wall-clock time for a block of code.

    Args:
        label: A descriptive label printed alongside the elapsed time.

    Example::

        with timer("Embedding generation"):
            generate_embeddings(words)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{label}: {elapsed:.2f}s")


def ensure_dir(path: Path) -> Path:
    """Create *path* and all intermediate parents if they do not exist.

    Args:
        path: Directory path to create.

    Returns:
        The same *path* for chaining convenience.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def word_to_index(words: list[str]) -> dict[str, int]:
    """Build a word-to-index lookup dictionary from an ordered word list.

    Args:
        words: Ordered list of words (aligned with an embedding matrix).

    Returns:
        A mapping ``{word: index}`` for O(1) lookups.
    """
    return {word: idx for idx, word in enumerate(words)}

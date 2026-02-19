"""Storage module for persisting and loading embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
WORDS_PATH = DATA_DIR / "words.txt"


def save_embeddings(
    embeddings: np.ndarray,
    words: list[str],
    embeddings_path: Path = EMBEDDINGS_PATH,
    words_path: Path = WORDS_PATH,
) -> None:
    """Persist embeddings and the aligned word list to disk.

    Args:
        embeddings: NumPy array of shape ``(N, D)``.
        words: Ordered list of N words aligned with *embeddings*.
        embeddings_path: Destination path for the ``.npy`` file.
        words_path: Destination path for the word list text file.
    """
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    words_path.write_text("\n".join(words), encoding="utf-8")


def load_embeddings(
    embeddings_path: Path = EMBEDDINGS_PATH,
    words_path: Path = WORDS_PATH,
) -> tuple[np.ndarray, list[str]]:
    """Load embeddings and the aligned word list from disk.

    Args:
        embeddings_path: Path to the ``.npy`` embeddings file.
        words_path: Path to the word list text file.

    Returns:
        A tuple ``(embeddings, words)`` where *embeddings* is a float32
        NumPy array of shape ``(N, D)`` and *words* is a list of N strings.

    Raises:
        FileNotFoundError: If either file does not exist.
    """
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not words_path.exists():
        raise FileNotFoundError(f"Words file not found: {words_path}")

    embeddings: np.ndarray = np.load(embeddings_path)
    words = words_path.read_text(encoding="utf-8").splitlines()
    return embeddings, words


def embeddings_exist(
    embeddings_path: Path = EMBEDDINGS_PATH,
    words_path: Path = WORDS_PATH,
) -> bool:
    """Return ``True`` if both persisted embedding files already exist.

    Args:
        embeddings_path: Path to the ``.npy`` embeddings file.
        words_path: Path to the word list text file.
    """
    return embeddings_path.exists() and words_path.exists()

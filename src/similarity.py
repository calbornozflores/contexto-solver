"""Similarity module for semantic nearest-neighbour search."""

from __future__ import annotations

import numpy as np

from src.embedding_model import generate_embeddings, load_model
from src.storage import load_embeddings


def most_similar(
    query: str,
    top_k: int = 10,
    embeddings: np.ndarray | None = None,
    words: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Find the most semantically similar words to *query*.

    Uses cosine similarity (dot product on L2-normalised vectors) for fast
    vectorised computation.

    If *embeddings* and *words* are not provided they are loaded from disk via
    :func:`src.storage.load_embeddings`.

    Args:
        query: The query word or phrase to search for.
        top_k: Number of top results to return.
        embeddings: Pre-loaded embedding matrix of shape ``(N, D)``.  When
            ``None`` the embeddings are loaded from disk.
        words: Word list aligned with *embeddings*.  When ``None`` the word
            list is loaded from disk.

    Returns:
        A list of ``(word, similarity_score)`` tuples sorted by descending
        similarity (most similar first).
    """
    if embeddings is None or words is None:
        embeddings, words = load_embeddings()

    # Embed and normalise the query vector.
    query_emb = generate_embeddings([query])  # shape (1, D), already normalised
    query_vec: np.ndarray = query_emb[0]

    # Cosine similarities via dot product (embeddings are already normalised).
    scores: np.ndarray = embeddings @ query_vec  # shape (N,)

    # Retrieve top_k indices (unsorted), then sort.
    top_k = min(top_k, len(words))
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return [(words[i], float(scores[i])) for i in top_indices]

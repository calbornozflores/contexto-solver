"""Embedding model module using sentence-transformers."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
_DEFAULT_BATCH_SIZE = 512

# Module-level cache so the model is loaded only once per process.
_model_cache: dict[str, "SentenceTransformer"] = {}  # type: ignore[name-defined]


def load_model(model_name: str = _DEFAULT_MODEL_NAME):
    """Load (or retrieve from cache) a SentenceTransformer model.

    Args:
        model_name: Name of the sentence-transformers model to load.

    Returns:
        A loaded ``SentenceTransformer`` instance.
    """
    # Import here so the rest of the module can be imported without
    # sentence-transformers installed (useful for testing stubs).
    from sentence_transformers import SentenceTransformer  # type: ignore

    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def generate_embeddings(
    words: list[str],
    model_name: str = _DEFAULT_MODEL_NAME,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Generate embeddings for a list of words using the local transformer model.

    Embeddings are L2-normalised so that cosine similarity reduces to a dot
    product, enabling fast vectorised comparisons.

    Args:
        words: Ordered list of words to embed.
        model_name: Name of the sentence-transformers model to use.
        batch_size: Number of words to process in each batch.

    Returns:
        Float32 NumPy array of shape ``(len(words), embedding_dim)``.
    """
    model = load_model(model_name)
    all_embeddings: list[np.ndarray] = []

    for start in tqdm(range(0, len(words), batch_size), desc="Generating embeddings"):
        batch = words[start : start + batch_size]
        batch_emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(batch_emb.astype(np.float32))

    embeddings = np.vstack(all_embeddings)
    # L2-normalise for cosine similarity via dot product.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
    embeddings /= norms
    return embeddings

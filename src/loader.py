"""Loader module for reading the words dictionary."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
WORDS_DICTIONARY_PATH = DATA_DIR / "words_dictionary.json"


def load_words(path: Path = WORDS_DICTIONARY_PATH) -> set[str]:
    """Load English words from words_dictionary.json into a set for fast lookup.

    Args:
        path: Path to the JSON file. Keys are words, values are 1.

    Returns:
        A set of lowercase English words.
    """
    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, int] = json.load(f)
    return set(data.keys())

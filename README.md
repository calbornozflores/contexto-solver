# contexto-solver

A fully-local semantic word similarity system built with [sentence-transformers](https://www.sbert.net/).
Precompute embeddings once, then run fast cosine-similarity queries — perfect for solving word-similarity games like [Contexto](https://contexto.me/).

---

## Project structure

```
contexto-solver/
│
├── data/
│   ├── words_dictionary.json   ← word list (keys = words, values = 1)
│   ├── embeddings.npy          ← generated embedding matrix  (N × D)
│   └── words.txt               ← word list aligned with embeddings
│
├── src/
│   ├── loader.py               ← load_words()
│   ├── embedding_model.py      ← load_model(), generate_embeddings()
│   ├── storage.py              ← save_embeddings(), load_embeddings()
│   ├── similarity.py           ← most_similar()
│   └── utils.py                ← timer(), ensure_dir(), word_to_index()
│
├── scripts/
│   └── generate_embeddings.py  ← CLI: precompute & persist embeddings
│
├── main.py                     ← interactive similarity query REPL
└── requirements.txt
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Provide / extend the word list

`data/words_dictionary.json` ships with a small set of common English words.
You can replace it with the full ~370 k-word list from
[dwyl/english-words](https://github.com/dwyl/english-words) or any other
JSON file whose keys are words and values are `1`.

### 3. Generate embeddings (once)

```bash
python scripts/generate_embeddings.py
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `all-MiniLM-L6-v2` | sentence-transformers model name |
| `--batch-size` | `512` | words per batch |
| `--force` | — | regenerate even if files already exist |

### 4. Run interactive queries

```bash
python main.py
```

```
Loading embeddings …
  Loaded 250 words, embedding dim=384.

Contexto Solver — semantic similarity search
Type a word to find the most similar words (Ctrl-C to quit).

Query: ocean
Top-10 most similar to 'ocean':
   1. water               0.6831
   2. fish                0.6204
   3. earth               0.5912
   ...
```

---

## API reference

### `src.loader`

```python
load_words(path: Path = ...) -> set[str]
```

### `src.embedding_model`

```python
load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer
generate_embeddings(words: list[str], model_name: str = ..., batch_size: int = 512) -> np.ndarray
```

### `src.storage`

```python
save_embeddings(embeddings: np.ndarray, words: list[str]) -> None
load_embeddings() -> tuple[np.ndarray, list[str]]
embeddings_exist() -> bool
```

### `src.similarity`

```python
most_similar(query: str, top_k: int = 10, embeddings=None, words=None) -> list[tuple[str, float]]
```

---

## Performance notes

* Embeddings are **L2-normalised** at generation time, so cosine similarity is
  computed as a single matrix–vector dot product — O(N × D) with NumPy BLAS.
* Embeddings are only generated once; subsequent runs load from `.npy` files in
  milliseconds.
* For vocabularies > 500 k words you can optionally integrate
  [FAISS](https://github.com/facebookresearch/faiss) by replacing the dot-product
  loop in `src/similarity.py` with a FAISS flat index.

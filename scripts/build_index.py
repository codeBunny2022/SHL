#!/usr/bin/env python3
"""Build embedding index for the normalized catalog."""

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import CATALOG_PATH, DATA_DIR, EMBEDDINGS_PATH, GEMINI_API_KEY, GEMINI_EMBED_MODEL

BATCH_SIZE = 20


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9+#]+", text.lower())


def build_tfidf_embeddings(texts: list[str]) -> tuple[np.ndarray, list[str], dict[str, int], dict[str, int]]:
    """Local fallback when Gemini API key is unavailable."""
    doc_tokens = [_tokenize(t) for t in texts]
    df: Counter[str] = Counter()
    for tokens in doc_tokens:
        df.update(set(tokens))
    n_docs = len(texts)
    vocab = sorted(df)
    vocab_index = {w: i for i, w in enumerate(vocab)}
    matrix = np.zeros((n_docs, len(vocab)), dtype=np.float32)
    for row, tokens in enumerate(doc_tokens):
        tf = Counter(tokens)
        for term, count in tf.items():
            if term not in vocab_index:
                continue
            idf = np.log((1 + n_docs) / (1 + df[term])) + 1
            matrix[row, vocab_index[term]] = count * idf
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-8), vocab, vocab_index, dict(df)


def embed_batch_gemini(client, texts: list[str]) -> list[list[float]]:
    result = client.models.embed_content(
        model=GEMINI_EMBED_MODEL,
        contents=texts,
    )
    return [e.values for e in result.embeddings]


def main() -> None:
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)

    texts = [item["embed_text"] for item in catalog]

    if GEMINI_API_KEY:
        from google import genai

        client = genai.Client(api_key=GEMINI_API_KEY)
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            vectors.extend(embed_batch_gemini(client, batch))
            print(f"Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")
            time.sleep(0.2)
        matrix = np.array(vectors, dtype=np.float32)
        with open(DATA_DIR / "index_meta.json", "w", encoding="utf-8") as f:
            json.dump({"type": "gemini"}, f)
        print("Used Gemini embeddings")
    else:
        matrix, vocab, vocab_index, df = build_tfidf_embeddings(texts)
        meta = {
            "type": "tfidf",
            "vocab": vocab,
            "df": df,
            "n_docs": len(texts),
        }
        with open(DATA_DIR / "index_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        print("Used local TF-IDF fallback (set GEMINI_API_KEY for Gemini embeddings)")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, matrix)
    print(f"Saved {matrix.shape} to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()

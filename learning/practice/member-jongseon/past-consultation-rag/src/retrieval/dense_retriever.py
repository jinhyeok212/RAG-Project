from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class DenseExactRetriever:
    """Lightweight dense-like bi-encoder fallback using char/word TF-IDF vectors.

    If sentence-transformers is installed, this class can be replaced later without
    changing downstream result schemas.
    """

    def __init__(self, max_features: int = 80000, char_ngram_min: int = 2, char_ngram_max: int = 4) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(char_ngram_min, char_ngram_max),
            max_features=max_features,
            lowercase=True,
        )
        self.doc_ids: list[str] = []
        self.matrix = None

    def fit(self, doc_ids: list[str], texts: list[str]) -> "DenseExactRetriever":
        self.doc_ids = doc_ids
        self.matrix = normalize(self.vectorizer.fit_transform(texts), norm="l2", axis=1)
        return self

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if self.matrix is None or not self.doc_ids:
            return []
        q = normalize(self.vectorizer.transform([query]), norm="l2", axis=1)
        scores = (self.matrix @ q.T).toarray().ravel()
        top_idx = np.argsort(-scores)[:top_k]
        return [
            {"doc_id": self.doc_ids[int(idx)], "dense_rank": rank + 1, "dense_score": float(scores[int(idx)])}
            for rank, idx in enumerate(top_idx)
            if scores[int(idx)] > 0
        ]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "DenseExactRetriever":
        with open(path, "rb") as f:
            return pickle.load(f)

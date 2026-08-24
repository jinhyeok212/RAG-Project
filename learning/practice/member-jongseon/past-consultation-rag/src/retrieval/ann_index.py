from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from src.retrieval.dense_retriever import DenseExactRetriever


class DenseANNRetriever:
    """Approximate-style index over reduced dense vectors using sklearn neighbors."""

    def __init__(self, exact: DenseExactRetriever, svd_dim: int = 128, n_neighbors: int = 50) -> None:
        self.exact = exact
        self.svd_dim = svd_dim
        self.n_neighbors = n_neighbors
        self.svd: TruncatedSVD | None = None
        self.nn: NearestNeighbors | None = None
        self.doc_ids = exact.doc_ids
        self.build_seconds = 0.0
        self.index_size_bytes = 0

    def fit(self) -> "DenseANNRetriever":
        if self.exact.matrix is None:
            raise ValueError("Exact dense retriever must be fitted first")
        start = time.perf_counter()
        dim = min(self.svd_dim, max(2, self.exact.matrix.shape[1] - 1), max(2, self.exact.matrix.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=dim, random_state=42)
        mat = self.svd.fit_transform(self.exact.matrix)
        mat = normalize(mat)
        self.nn = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(self.doc_ids)), metric="cosine", algorithm="auto")
        self.nn.fit(mat)
        self._mat = mat
        self.build_seconds = time.perf_counter() - start
        self.index_size_bytes = mat.nbytes
        return self

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if self.nn is None or self.svd is None:
            return []
        q = self.exact.vectorizer.transform([query])
        q = normalize(self.svd.transform(q))
        distances, indices = self.nn.kneighbors(q, n_neighbors=min(top_k, len(self.doc_ids)))
        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            score = 1.0 - float(dist)
            results.append({"doc_id": self.doc_ids[int(idx)], "dense_rank": rank, "dense_score": score})
        return results

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "DenseANNRetriever":
        with open(path, "rb") as f:
            return pickle.load(f)

from __future__ import annotations

import math
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from src.retrieval.tokenization import tokenize


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenizer: str = "mixed") -> None:
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.doc_ids: list[str] = []
        self.doc_tokens: list[Counter[str]] = []
        self.doc_lens: list[int] = []
        self.idf: dict[str, float] = {}
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.avgdl = 0.0

    def fit(self, doc_ids: list[str], texts: list[str]) -> "BM25Retriever":
        self.doc_ids = doc_ids
        dfs: Counter[str] = Counter()
        postings: dict[str, list[tuple[int, int]]] = {}
        for doc_idx, text in enumerate(texts):
            counts = Counter(tokenize(text, self.tokenizer))
            self.doc_tokens.append(counts)
            self.doc_lens.append(sum(counts.values()))
            dfs.update(counts.keys())
            for term, tf in counts.items():
                postings.setdefault(term, []).append((doc_idx, tf))
        n_docs = len(self.doc_ids)
        self.avgdl = float(np.mean(self.doc_lens)) if self.doc_lens else 0.0
        self.idf = {term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5)) for term, df in dfs.items()}
        self.postings = postings
        return self

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        q_terms = tokenize(query, self.tokenizer)
        scores: dict[int, float] = {}
        for term in q_terms:
            idf = self.idf.get(term, 0.0)
            if not idf:
                continue
            for i, tf in self.postings.get(term, []):
                dl = self.doc_lens[i] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[i] = scores.get(i, 0.0) + idf * (tf * (self.k1 + 1)) / denom
        if not scores:
            return []
        top_idx = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [
            {"doc_id": self.doc_ids[int(idx)], "bm25_rank": rank + 1, "bm25_score": float(scores[int(idx)])}
            for rank, idx in enumerate(top_idx)
            if scores[int(idx)] > 0
        ]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "BM25Retriever":
        with open(path, "rb") as f:
            return pickle.load(f)

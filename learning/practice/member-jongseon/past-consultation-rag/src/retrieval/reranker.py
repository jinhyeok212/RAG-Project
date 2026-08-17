from __future__ import annotations

from src.retrieval.tokenization import tokenize


class LexicalOverlapReranker:
    """CPU-safe fallback reranker.

    It scores question/document pairs by token overlap and keeps the same output
    contract as a Cross-Encoder reranker.
    """

    def __init__(self, tokenizer: str = "mixed") -> None:
        self.tokenizer = tokenizer

    def score(self, query: str, text: str) -> float:
        q = set(tokenize(query, self.tokenizer))
        d = set(tokenize(text, self.tokenizer))
        if not q or not d:
            return 0.0
        return len(q & d) / len(q | d)

    def rerank(self, query: str, candidates: list[dict], doc_texts: dict[str, str], top_k: int = 5) -> list[dict]:
        rows = []
        for item in candidates:
            row = dict(item)
            row["reranker_score"] = self.score(query, doc_texts.get(item["doc_id"], ""))
            rows.append(row)
        rows.sort(key=lambda x: (x["reranker_score"], x.get("rrf_score", 0.0)), reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["final_rank"] = rank
        return rows[:top_k]

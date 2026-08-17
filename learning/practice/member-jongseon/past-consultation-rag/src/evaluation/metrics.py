from __future__ import annotations

import math


def hit_at_k(results: list[str], positives: set[str], k: int) -> float:
    return 1.0 if any(doc_id in positives for doc_id in results[:k]) else 0.0


def recall_at_k(results: list[str], positives: set[str], k: int) -> float:
    if not positives:
        return 0.0
    return len(set(results[:k]) & positives) / len(positives)


def mrr(results: list[str], positives: set[str]) -> float:
    for idx, doc_id in enumerate(results, start=1):
        if doc_id in positives:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(results: list[str], positives: set[str], k: int) -> float:
    dcg = 0.0
    for idx, doc_id in enumerate(results[:k], start=1):
        if doc_id in positives:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(positives), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    return {key: sum(float(r.get(key, 0.0)) for r in rows) / len(rows) for key in keys}

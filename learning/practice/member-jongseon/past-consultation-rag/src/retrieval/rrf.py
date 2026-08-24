from __future__ import annotations


def reciprocal_rank_fusion(result_sets: list[list[dict]], k: int = 60, weights: list[float] | None = None) -> list[dict]:
    scores: dict[str, dict] = {}
    weights = weights or [1.0] * len(result_sets)
    for results, weight in zip(result_sets, weights):
        for rank, item in enumerate(results, start=1):
            doc_id = item["doc_id"]
            row = scores.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "bm25_rank": None,
                    "bm25_score": 0.0,
                    "dense_rank": None,
                    "dense_score": 0.0,
                    "rrf_score": 0.0,
                    "reranker_score": None,
                    "final_rank": None,
                },
            )
            row.update({key: value for key, value in item.items() if key != "doc_id" and value is not None})
            row["rrf_score"] += weight / (k + rank)
    fused = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    for rank, item in enumerate(fused, start=1):
        item["final_rank"] = rank
    return fused

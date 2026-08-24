from __future__ import annotations

from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.reranker import LexicalOverlapReranker


class HybridRetriever:
    def __init__(self, bm25=None, dense=None, ann=None, docs: dict[str, dict] | None = None, cfg: dict | None = None) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.ann = ann
        self.docs = docs or {}
        self.cfg = cfg or {}
        self.reranker = LexicalOverlapReranker()

    def search(
        self,
        query: str,
        top_k: int = 5,
        use_sparse: bool = True,
        use_dense: bool = True,
        use_ann: bool = False,
        use_reranker: bool = False,
        category: str | None = None,
        intent: str | None = None,
        candidate_k: int | None = None,
        filter_pool_k: int | None = None,
        rrf_k: int | None = None,
        sparse_weight: float | None = None,
        dense_weight: float | None = None,
    ) -> list[dict]:
        cfg_candidate_k = int(self.cfg.get("retrieval", {}).get("reranker", {}).get("candidate_k", 50))
        candidate_k = max(top_k, int(candidate_k or cfg_candidate_k))
        fetch_k = candidate_k
        if category or intent:
            fetch_k = min(len(self.docs) or candidate_k, max(candidate_k, int(filter_pool_k or 1000)))
        result_sets = []
        result_weights = []
        if use_sparse and self.bm25:
            result_sets.append(self.bm25.search(query, fetch_k))
            result_weights.append(float(sparse_weight if sparse_weight is not None else self.cfg.get("retrieval", {}).get("hybrid", {}).get("sparse_weight", 1.0)))
        if use_dense:
            dense_backend = self.ann if use_ann and self.ann else self.dense
            if dense_backend:
                result_sets.append(dense_backend.search(query, fetch_k))
                result_weights.append(float(dense_weight if dense_weight is not None else self.cfg.get("retrieval", {}).get("hybrid", {}).get("dense_weight", 1.0)))
        if not result_sets:
            return []
        if len(result_sets) == 1:
            fused = result_sets[0]
            for i, row in enumerate(fused, start=1):
                row.setdefault("rrf_score", 0.0)
                row.setdefault("final_rank", i)
        else:
            rrf_cfg = self.cfg.get("retrieval", {}).get("hybrid", {})
            fused = reciprocal_rank_fusion(result_sets, k=int(rrf_k or rrf_cfg.get("rrf_k", 60)), weights=result_weights)

        if category or intent:
            filtered = []
            for row in fused:
                doc = self.docs.get(row["doc_id"], {})
                if category and doc.get("category") != category:
                    continue
                if intent and doc.get("intent") != intent:
                    continue
                filtered.append(row)
            fused = filtered

        if use_reranker:
            doc_texts = {doc_id: doc.get("retrieval_text", "") for doc_id, doc in self.docs.items()}
            return self.reranker.rerank(query, fused[:candidate_k], doc_texts, top_k=top_k)
        return fused[:top_k]

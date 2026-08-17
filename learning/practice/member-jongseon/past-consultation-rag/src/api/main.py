from __future__ import annotations

from pathlib import Path

import joblib

from src.common import load_config
from src.generation.generator import generate_answer
from src.retrieval.ann_index import DenseANNRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseExactRetriever
from src.retrieval.hybrid_retriever import HybridRetriever

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    FastAPI = None
    BaseModel = object


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    category: str | None = None
    intent: str | None = None
    use_sparse: bool = True
    use_dense: bool = True
    use_ann: bool = False
    use_reranker: bool = True
    candidate_k: int | None = None
    filter_pool_k: int | None = None
    rrf_k: int | None = None
    sparse_weight: float | None = None
    dense_weight: float | None = None


if FastAPI is None:
    app = None
else:
    app = FastAPI(title="Past Consultation RAG")
    cfg = load_config("configs/mvp.yaml")
    index_dir = Path(cfg["paths"]["index_dir"])
    bm25 = BM25Retriever.load(index_dir / "bm25.pkl")
    dense = DenseExactRetriever.load(index_dir / "dense_exact.pkl")
    ann = DenseANNRetriever.load(index_dir / "dense_ann.pkl") if (index_dir / "dense_ann.pkl").exists() else None
    docs = joblib.load(index_dir / "docs.joblib")
    retriever = HybridRetriever(bm25=bm25, dense=dense, ann=ann, docs=docs, cfg=cfg)

    @app.get("/health")
    def health():
        return {"status": "ok", "documents": len(docs), "ann": ann is not None}

    @app.post("/search")
    def search(req: SearchRequest):
        results = retriever.search(
            req.query,
            top_k=req.top_k,
            category=req.category,
            intent=req.intent,
            use_sparse=req.use_sparse,
            use_dense=req.use_dense,
            use_ann=req.use_ann,
            use_reranker=req.use_reranker,
            candidate_k=req.candidate_k,
            filter_pool_k=req.filter_pool_k,
            rrf_k=req.rrf_k,
            sparse_weight=req.sparse_weight,
            dense_weight=req.dense_weight,
        )
        return {"query": req.query, "results": results}

    @app.post("/ask")
    def ask(req: SearchRequest):
        results = search(req)["results"]
        return generate_answer(req.query, results[: req.top_k], docs)

    @app.get("/metrics")
    def metrics():
        import pandas as pd

        path = Path(cfg["paths"]["report_dir"]) / "retrieval_metrics_summary.csv"
        if not path.exists():
            return {"message": "No metrics yet. Run evaluation first."}
        return {"summary": pd.read_csv(path).to_dict("records")}

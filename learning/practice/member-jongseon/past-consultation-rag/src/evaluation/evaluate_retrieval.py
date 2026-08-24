from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd

from src.common import ensure_dirs, load_config
from src.evaluation.metrics import aggregate, hit_at_k, mrr, ndcg_at_k, recall_at_k
from src.retrieval.ann_index import DenseANNRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseExactRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


def load_queries(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def eval_method(name: str, retriever: HybridRetriever, queries: list[dict], search_kwargs: dict, top_k: int) -> tuple[list[dict], list[dict]]:
    rows = []
    failures = []
    for q in queries:
        start = time.perf_counter()
        results = retriever.search(q["query"], top_k=max(10, top_k), **search_kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        ranked = [r["doc_id"] for r in results]
        positives = set(q["positive_doc_ids"])
        row = {
            "method": name,
            "query_id": q["query_id"],
            "category": q.get("category"),
            "intent": q.get("intent"),
            "query_length": len(q.get("query", "")),
            "hit@1": hit_at_k(ranked, positives, 1),
            "hit@3": hit_at_k(ranked, positives, 3),
            "hit@5": hit_at_k(ranked, positives, 5),
            "hit@10": hit_at_k(ranked, positives, 10),
            "recall@1": recall_at_k(ranked, positives, 1),
            "recall@5": recall_at_k(ranked, positives, 5),
            "recall@10": recall_at_k(ranked, positives, 10),
            "mrr": mrr(ranked, positives),
            "ndcg@10": ndcg_at_k(ranked, positives, 10),
            "latency_ms": elapsed_ms,
        }
        rows.append(row)
        if row["hit@5"] == 0 and len(failures) < 200:
            failures.append(
                {
                    "method": name,
                    "query_id": q["query_id"],
                    "query": q["query"],
                    "positive_doc_ids": q["positive_doc_ids"],
                    "top_results": results[:5],
                }
            )
    return rows, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    processed = Path(cfg["paths"]["processed_dir"])
    index_dir = Path(cfg["paths"]["index_dir"])
    report_dir = Path(cfg["paths"]["report_dir"])
    ensure_dirs(report_dir)

    bm25 = BM25Retriever.load(index_dir / "bm25.pkl")
    dense = DenseExactRetriever.load(index_dir / "dense_exact.pkl")
    ann = DenseANNRetriever.load(index_dir / "dense_ann.pkl") if (index_dir / "dense_ann.pkl").exists() else None
    docs = joblib.load(index_dir / "docs.joblib")
    retriever = HybridRetriever(bm25=bm25, dense=dense, ann=ann, docs=docs, cfg=cfg)
    queries = load_queries(processed / "validation_retrieval_queries.jsonl", args.limit)
    top_k = int(cfg["retrieval"].get("top_k", 10))

    methods = [
        ("BM25", {"use_sparse": True, "use_dense": False, "use_ann": False, "use_reranker": False}),
        ("Dense Exact", {"use_sparse": False, "use_dense": True, "use_ann": False, "use_reranker": False}),
        ("Dense ANN", {"use_sparse": False, "use_dense": True, "use_ann": True, "use_reranker": False}),
        ("Hybrid RRF", {"use_sparse": True, "use_dense": True, "use_ann": False, "use_reranker": False}),
        ("Hybrid + Reranker", {"use_sparse": True, "use_dense": True, "use_ann": False, "use_reranker": True}),
    ]
    all_rows = []
    all_failures = []
    for name, kwargs in methods:
        if name == "Dense ANN" and ann is None:
            continue
        rows, failures = eval_method(name, retriever, queries, kwargs, top_k)
        all_rows.extend(rows)
        all_failures.extend(failures)

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(report_dir / "retrieval_metrics.csv", index=False, encoding="utf-8-sig")
    summary = metrics_df.groupby("method", as_index=False).agg(
        **{
            "Hit@5": ("hit@5", "mean"),
            "Recall@5": ("recall@5", "mean"),
            "MRR": ("mrr", "mean"),
            "nDCG@10": ("ndcg@10", "mean"),
            "평균 지연시간(ms)": ("latency_ms", "mean"),
            "p95 지연시간(ms)": ("latency_ms", lambda s: s.quantile(0.95)),
        }
    )
    summary.to_csv(report_dir / "retrieval_metrics_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_failures).to_csv(report_dir / "failure_cases.csv", index=False, encoding="utf-8-sig")
    by_cat = metrics_df.groupby(["method", "category"], as_index=False)[["hit@5", "mrr", "ndcg@10", "latency_ms"]].mean()
    by_cat.to_csv(report_dir / "category_metrics.csv", index=False, encoding="utf-8-sig")
    by_intent = metrics_df.groupby(["method", "intent"], as_index=False)[["hit@5", "mrr", "ndcg@10", "latency_ms"]].mean()
    by_intent.to_csv(report_dir / "intent_metrics.csv", index=False, encoding="utf-8-sig")
    ann_rows = []
    if ann:
        exact_rows = metrics_df[metrics_df["method"] == "Dense Exact"].set_index("query_id")
        ann_rows_df = metrics_df[metrics_df["method"] == "Dense ANN"].set_index("query_id")
        ann_rows.append(
            {
                "ANN Recall@10 vs exact proxy": float((ann_rows_df["hit@10"] == exact_rows["hit@10"]).mean()) if len(ann_rows_df) else 0.0,
                "index_build_seconds": ann.build_seconds,
                "index_size_bytes": ann.index_size_bytes,
            }
        )
    pd.DataFrame(ann_rows).to_csv(report_dir / "ann_benchmark.csv", index=False, encoding="utf-8-sig")

    md = ["# 검색 성능 평가", "", "| 방식 | Hit@5 | Recall@5 | MRR | nDCG@10 | 평균 지연시간(ms) | p95 지연시간(ms) |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.to_dict("records"):
        md.append(
            f"| {row['method']} | {row['Hit@5']:.4f} | {row['Recall@5']:.4f} | {row['MRR']:.4f} | {row['nDCG@10']:.4f} | {row['평균 지연시간(ms)']:.2f} | {row['p95 지연시간(ms)']:.2f} |"
        )
    md.append("")
    md.append("## 해석")
    md.append("- Validation 답변 문서는 Train 검색 corpus에 넣지 않고, 같은 질문에 대응되는 Train 과거 상담 사례를 찾는 방식으로 평가했습니다.")
    md.append("- 현재 Dense는 sentence-transformers 미설치 환경을 고려한 sklearn TF-IDF bi-encoder fallback입니다.")
    md.append("- Reranker는 CPU-safe lexical-overlap fallback이며, Cross-Encoder 모델 설치 후 교체할 수 있습니다.")
    (report_dir / "retrieval_metrics.md").write_text("\n".join(md), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()

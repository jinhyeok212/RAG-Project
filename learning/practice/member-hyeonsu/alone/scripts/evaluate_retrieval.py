"""Validation 500개 질문으로 Retrieval-only 검색을 평가합니다."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.chroma_store import ChromaQuestionStore
from src.embedding_model import KoreanEmbeddingModel
from src.retrieval_evaluator import classify_failure_candidate, evaluate_query, normalize_intent
from src.retrieval_utils import env_int, load_environment, read_and_validate_csv


def main() -> None:
    """질문별 결과, 요약, 전체/인텐트별 지표와 실패 후보를 저장합니다."""
    load_environment()
    input_path = PROJECT_ROOT / "data" / "experiment" / "super_validation_sample_500.csv"
    output_dir = PROJECT_ROOT / "results" / "retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)
    required = [
        "query_id", "question", "reference_answer", "expected_intent",
        "category", "original_document_id", "conversation_id", "qa_number",
    ]
    frame = read_and_validate_csv(input_path, required)
    if len(frame) != 500:
        raise ValueError(f"Validation 표본은 500개여야 합니다. 실제: {len(frame)}")

    # 전체 시간에는 Chroma 및 모델 로드, 임베딩, 검색, 결과 집계를 모두 포함합니다.
    total_started = time.perf_counter()
    top_k = env_int("TOP_K", 5)
    evaluation_depth = max(top_k, env_int("EVALUATION_DEPTH", 20))
    batch_size = env_int("EMBEDDING_BATCH_SIZE", 64)
    store = ChromaQuestionStore()
    if store.count == 0:
        raise RuntimeError("Chroma 컬렉션이 비어 있습니다. 먼저 인덱스를 생성하세요.")
    model = KoreanEmbeddingModel()

    embedding_started = time.perf_counter()
    query_embeddings = model.encode(frame["question"].tolist(), batch_size=batch_size)
    embedding_total = time.perf_counter() - embedding_started
    embedding_ms = embedding_total * 1000 / len(frame)

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for position, row in enumerate(frame.itertuples(index=False)):
        search_started = time.perf_counter()
        results = store.query(query_embeddings[position], evaluation_depth)
        search_ms = (time.perf_counter() - search_started) * 1000
        expected = normalize_intent(row.expected_intent)
        metrics = evaluate_query(expected, results)

        for result in results:
            metadata = result["metadata"]
            retrieved_intent = normalize_intent(metadata.get("intent"))
            detail_rows.append({
                "query_id": row.query_id,
                "query": row.question,
                "expected_intent": expected,
                "rank": result["rank"],
                "retrieved_document_id": result["document_id"],
                "retrieved_question": result["question"],
                "retrieved_answer": metadata.get("answer", ""),
                "retrieved_intent": retrieved_intent,
                "distance": result["distance"],
                "intent_match": int(expected != "" and expected == retrieved_intent),
            })
        summary_rows.append({
            "query_id": row.query_id,
            "query": row.question,
            "expected_intent": expected,
            "first_matching_rank": metrics.first_matching_rank,
            "hit_at_1": metrics.hit_at_1,
            "hit_at_3": metrics.hit_at_3,
            "hit_at_5": metrics.hit_at_5,
            "reciprocal_rank": metrics.reciprocal_rank,
            "query_embedding_time_ms": embedding_ms,
            "search_time_ms": search_ms,
        })
        if metrics.hit_at_5 == 0 or (
            metrics.first_matching_rank is not None and metrics.first_matching_rank > 5
        ):
            top1 = results[0] if results else None
            top1_metadata = top1["metadata"] if top1 else {}
            top1_intent = normalize_intent(top1_metadata.get("intent"))
            failure_type, review_note = classify_failure_candidate(
                row.question, expected, top1_intent, metrics.first_matching_rank
            )
            failure_rows.append({
                "query": row.question,
                "expected_intent": expected,
                "top1_question": "" if top1 is None else top1["question"],
                "top1_intent": top1_intent,
                "top1_distance": "" if top1 is None else top1["distance"],
                "first_matching_rank": metrics.first_matching_rank,
                "failure_type_candidate": failure_type,
                "review_note": review_note,
            })

    total_elapsed = time.perf_counter() - total_started
    details = pd.DataFrame(detail_rows)
    summaries = pd.DataFrame(summary_rows)
    failures = pd.DataFrame(failure_rows)
    details.to_csv(output_dir / "super_validation_retrieval_results.csv", index=False, encoding="utf-8-sig")
    summaries.to_csv(output_dir / "super_validation_query_summary.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(output_dir / "super_retrieval_failure_candidates.csv", index=False, encoding="utf-8-sig")

    overall = pd.DataFrame([{
        "model_name": model.model_name,
        "collection": store.collection_name,
        "training_document_count": store.count,
        "validation_query_count": len(summaries),
        "intent_hit_at_1": summaries["hit_at_1"].mean(),
        "intent_hit_at_3": summaries["hit_at_3"].mean(),
        "intent_hit_at_5": summaries["hit_at_5"].mean(),
        "intent_mrr": summaries["reciprocal_rank"].mean(),
        "average_query_embedding_time_ms": summaries["query_embedding_time_ms"].mean(),
        "average_search_time_ms": summaries["search_time_ms"].mean(),
        "total_evaluation_time_seconds": total_elapsed,
        "evaluation_depth": evaluation_depth,
    }])
    overall.to_csv(output_dir / "super_retrieval_metrics.csv", index=False, encoding="utf-8-sig")

    intent_metrics = summaries.groupby("expected_intent", as_index=False).agg(
        validation_query_count=("query_id", "count"),
        intent_hit_at_1=("hit_at_1", "mean"),
        intent_hit_at_3=("hit_at_3", "mean"),
        intent_hit_at_5=("hit_at_5", "mean"),
        intent_mrr=("reciprocal_rank", "mean"),
        average_search_time_ms=("search_time_ms", "mean"),
    )
    intent_metrics.to_csv(output_dir / "super_intent_metrics.csv", index=False, encoding="utf-8-sig")

    metric = overall.iloc[0]
    print(f"모델명: {model.model_name}")
    print(f"Validation 질문 수: {len(summaries)}")
    print(f"Intent Hit@1: {metric['intent_hit_at_1']:.6f}")
    print(f"Intent Hit@3: {metric['intent_hit_at_3']:.6f}")
    print(f"Intent Hit@5: {metric['intent_hit_at_5']:.6f}")
    print(f"Intent MRR: {metric['intent_mrr']:.6f}")
    print(f"평균 질문 임베딩 시간: {metric['average_query_embedding_time_ms']:.3f}ms")
    print(f"평균 검색 시간: {metric['average_search_time_ms']:.3f}ms")
    print(f"전체 평가 시간: {metric['total_evaluation_time_seconds']:.3f}초")
    print(f"상세 결과 저장 위치: {output_dir}")


if __name__ == "__main__":
    main()

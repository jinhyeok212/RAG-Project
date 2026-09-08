from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import chromadb
import joblib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data/processed/ontong_youth_chunks_v1/chunks.jsonl"
DEFAULT_EVAL = ROOT / "data/processed/ontong_youth_chunks_v1/eval_questions_with_gold_chunks.jsonl"
DEFAULT_INDEX_DIR = Path.home() / "chroma_indexes/ontong_youth_chroma_v1"
DEFAULT_OUTPUT_DIR = ROOT / "data/processed/ontong_youth_chunks_v1/evaluation"
DEFAULT_COLLECTION = "ontong_youth_chunks_v1_sklearn_tfidf_svd"
KST = timezone(timedelta(hours=9))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def hit_at_k(results: list[str], positives: set[str], k: int) -> float:
    return 1.0 if set(results[:k]) & positives else 0.0


def recall_at_k(results: list[str], positives: set[str], k: int) -> float:
    if not positives:
        return 0.0
    return len(set(results[:k]) & positives) / len(positives)


def mrr(results: list[str], positives: set[str]) -> float:
    for rank, item_id in enumerate(results, start=1):
        if item_id in positives:
            return 1.0 / rank
    return 0.0


def metrics(
    ranked_chunk_ids: list[str],
    gold_chunk_ids: set[str],
    chunk_to_document: dict[str, str],
    gold_document_ids: set[str],
) -> dict[str, float]:
    ranked_document_ids = [chunk_to_document[chunk_id] for chunk_id in ranked_chunk_ids]
    return {
        "chunk_hit@1": hit_at_k(ranked_chunk_ids, gold_chunk_ids, 1),
        "chunk_hit@3": hit_at_k(ranked_chunk_ids, gold_chunk_ids, 3),
        "chunk_hit@5": hit_at_k(ranked_chunk_ids, gold_chunk_ids, 5),
        "chunk_hit@10": hit_at_k(ranked_chunk_ids, gold_chunk_ids, 10),
        "chunk_recall@5": recall_at_k(ranked_chunk_ids, gold_chunk_ids, 5),
        "chunk_recall@10": recall_at_k(ranked_chunk_ids, gold_chunk_ids, 10),
        "chunk_mrr": mrr(ranked_chunk_ids, gold_chunk_ids),
        "document_hit@1": hit_at_k(ranked_document_ids, gold_document_ids, 1),
        "document_hit@3": hit_at_k(ranked_document_ids, gold_document_ids, 3),
        "document_hit@5": hit_at_k(ranked_document_ids, gold_document_ids, 5),
        "document_hit@10": hit_at_k(ranked_document_ids, gold_document_ids, 10),
        "document_mrr": mrr(ranked_document_ids, gold_document_ids),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "chunk_hit@1",
        "chunk_hit@3",
        "chunk_hit@5",
        "chunk_hit@10",
        "chunk_recall@5",
        "chunk_recall@10",
        "chunk_mrr",
        "document_hit@1",
        "document_hit@3",
        "document_hit@5",
        "document_hit@10",
        "document_mrr",
        "latency_ms",
    ]
    return {
        "method": "Chroma SVD",
        "query_count": len(rows),
        **{metric: round(mean(float(row[metric]) for row in rows), 6) for metric in metric_names},
    }


def compact_top_results(ids: list[str], distances: list[float], chunks_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for rank, (chunk_id, distance) in enumerate(zip(ids, distances), start=1):
        chunk = chunks_by_id[chunk_id]
        results.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "document_id": chunk["document_id"],
                "distance": round(float(distance), 6),
                "score": round(1.0 - float(distance), 6),
                "title": chunk.get("title", ""),
                "text_preview": " ".join(chunk.get("text", "").split())[:220],
            }
        )
    return results


def make_report(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Ontong Youth Chroma Evaluation v1",
            "",
            f"- created_at: {manifest['created_at']}",
            f"- collection: {manifest['collection']['name']}",
            f"- collection_count: {manifest['collection']['count']}",
            f"- embedding_backend: {manifest['embedding']['backend']}",
            "",
            "| method | chunk Hit@1 | chunk Hit@3 | chunk Hit@5 | chunk MRR | document Hit@1 | document Hit@5 | document MRR | latency ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            "| {method} | {chunk_hit@1:.4f} | {chunk_hit@3:.4f} | {chunk_hit@5:.4f} | {chunk_mrr:.4f} | {document_hit@1:.4f} | {document_hit@5:.4f} | {document_mrr:.4f} | {latency_ms:.2f} |".format(
                **summary
            ),
            "",
        ]
    )


def run(
    *,
    chunks_path: Path,
    eval_path: Path,
    index_dir: Path,
    output_dir: Path,
    collection_name: str,
    top_k: int,
) -> dict[str, Any]:
    chunks = read_jsonl(chunks_path)
    eval_questions = read_jsonl(eval_path)
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    chunk_to_document = {chunk["chunk_id"]: chunk["document_id"] for chunk in chunks}

    encoder_bundle = joblib.load(index_dir / "sklearn_tfidf_svd_encoder.joblib")
    encoder = encoder_bundle["encoder"]
    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    collection = client.get_collection(collection_name)

    metric_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for question in eval_questions:
        query = question["query"]
        started = time.perf_counter()
        query_embedding = encoder.transform([query]).astype("float32").tolist()[0]
        response = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["distances", "documents", "metadatas"],
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        ranked_chunk_ids = response["ids"][0]
        top_results = compact_top_results(
            response["ids"][0],
            response["distances"][0],
            chunks_by_id,
        )
        gold_chunk_ids = set(question.get("gold_chunk_ids") or [])
        gold_document_ids = set(question.get("ground_truth_document_ids") or [])
        row_metrics = metrics(ranked_chunk_ids, gold_chunk_ids, chunk_to_document, gold_document_ids)
        metric_row = {
            "method": "Chroma SVD",
            "query_id": question["query_id"],
            "question_type": question.get("question_type", ""),
            "primary_category": question.get("primary_category", ""),
            "primary_mclsf": question.get("primary_mclsf", ""),
            "gold_document_count": len(gold_document_ids),
            "gold_chunk_count": len(gold_chunk_ids),
            "retrieved_count": len(ranked_chunk_ids),
            "latency_ms": elapsed_ms,
            **row_metrics,
        }
        result_row = {
            "method": "Chroma SVD",
            "query_id": question["query_id"],
            "query": query,
            "ground_truth_document_ids": list(gold_document_ids),
            "gold_chunk_ids": list(gold_chunk_ids),
            "top_results": top_results,
            **row_metrics,
        }
        metric_rows.append(metric_row)
        result_rows.append(result_row)
        if row_metrics["document_hit@5"] == 0:
            failure_rows.append(result_row)

    summary = summarize(metric_rows)
    manifest = {
        "version": "ontong_youth_chroma_eval_v1",
        "created_at": datetime.now(KST).isoformat(timespec="seconds"),
        "input": {
            "chunks_path": str(chunks_path),
            "eval_questions_with_gold_chunks_path": str(eval_path),
            "index_dir": str(index_dir),
        },
        "output": {
            "metrics_path": str(output_dir / "chroma_retrieval_metrics.csv"),
            "summary_path": str(output_dir / "chroma_retrieval_summary.csv"),
            "top_results_path": str(output_dir / "chroma_retrieval_top_results.jsonl"),
            "failure_cases_path": str(output_dir / "chroma_retrieval_failure_cases.jsonl"),
            "report_path": str(output_dir / "chroma_retrieval_report.md"),
        },
        "collection": {
            "name": collection_name,
            "count": collection.count(),
        },
        "embedding": {
            "backend": encoder_bundle.get("embedding_backend", "sklearn_char_tfidf_svd"),
            "svd_dim": encoder_bundle.get("svd_dim"),
        },
        "retrieval": {
            "top_k": top_k,
        },
        "counts": {
            "eval_question_count": len(eval_questions),
            "failure_case_count": len(failure_rows),
        },
        "summary": summary,
    }

    write_csv(output_dir / "chroma_retrieval_metrics.csv", metric_rows)
    write_csv(output_dir / "chroma_retrieval_summary.csv", [summary])
    write_jsonl(output_dir / "chroma_retrieval_top_results.jsonl", result_rows)
    write_jsonl(output_dir / "chroma_retrieval_failure_cases.jsonl", failure_rows)
    write_json(output_dir / "chroma_retrieval_eval_manifest.json", manifest)
    (output_dir / "chroma_retrieval_report.md").write_text(
        make_report(summary, manifest),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Ontong Youth Chroma index v1.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run(
        chunks_path=args.chunks,
        eval_path=args.eval,
        index_dir=args.index_dir,
        output_dir=args.output_dir,
        collection_name=args.collection,
        top_k=args.top_k,
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

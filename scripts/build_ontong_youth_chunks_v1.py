from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS = ROOT / "data/processed/ontong_youth_mvp_400_documents/documents.jsonl"
DEFAULT_EVAL_QUESTIONS = ROOT / "data/processed/ontong_youth_eval_questions_v2_final_47/eval_questions.jsonl"
DEFAULT_QRELS = ROOT / "data/processed/ontong_youth_eval_questions_v2_final_47/qrels.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data/processed/ontong_youth_chunks_v1"

KST = timezone(timedelta(hours=9))
VERSION = "ontong_youth_chunks_v1"


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def percentile(values: list[int], pct: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * pct / 100) - 1))
    return ordered[index]


def chunk_spans(text_length: int, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    if text_length <= 0:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be greater than or equal to 0 and less than chunk_size")

    spans = []
    step = chunk_size - chunk_overlap
    start = 0
    while start < text_length:
        end = min(start + chunk_size, text_length)
        spans.append((start, end))
        if end >= text_length:
            break
        start += step
    return spans


def scalar_metadata(document: dict[str, Any]) -> dict[str, Any]:
    metadata = document.get("metadata") or {}
    return {
        "source_policy_no": metadata.get("source_policy_no", ""),
        "primary_category": metadata.get("primary_category", ""),
        "primary_mclsf": metadata.get("primary_mclsf", ""),
        "supervising_agency": metadata.get("supervising_agency", ""),
        "operating_agency": metadata.get("operating_agency", ""),
        "application_period": metadata.get("application_period", ""),
        "business_start_date": metadata.get("business_start_date", ""),
        "business_end_date": metadata.get("business_end_date", ""),
        "target_min_age": metadata.get("target_min_age", ""),
        "target_max_age": metadata.get("target_max_age", ""),
        "target_age_limit_yn": metadata.get("target_age_limit_yn", ""),
        "income_condition_code": metadata.get("income_condition_code", ""),
        "region_zip_codes": metadata.get("region_zip_codes", ""),
        "application_url": metadata.get("application_url", ""),
        "reference_url_1": metadata.get("reference_url_1", ""),
        "normalization_status": metadata.get("normalization_status", ""),
        "document_quality_score": metadata.get("document_quality_score", 0),
    }


def build_chunks(
    documents: list[dict[str, Any]],
    *,
    version: str,
    text_field: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for document in documents:
        document_id = document.get("document_id", "")
        text = document.get(text_field) or ""
        spans = chunk_spans(len(text), chunk_size, chunk_overlap)
        for chunk_index, (start_char, end_char) in enumerate(spans):
            chunk_text = text[start_char:end_char]
            chunk_id = f"{document_id}__chunk_{chunk_index:03d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "start_char": start_char,
                    "end_char": end_char,
                    "char_length": len(chunk_text),
                    "document_text_length": len(text),
                    "is_first_chunk": chunk_index == 0,
                    "is_last_chunk": chunk_index == len(spans) - 1,
                    "text_hash": short_hash(chunk_text),
                    "title": document.get("title", ""),
                    "source": document.get("source", ""),
                    "url": document.get("url", ""),
                    "metadata": {
                        "chunking_version": version,
                        "text_field": text_field,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        **scalar_metadata(document),
                    },
                }
            )
    return chunks


def qrels_by_query(qrels: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in qrels:
        if int(row.get("relevance") or 0) <= 0:
            continue
        query_id = row.get("query_id", "")
        document_id = row.get("document_id", "")
        key = (query_id, document_id)
        if not query_id or not document_id or key in seen:
            continue
        grouped[query_id].append(document_id)
        seen.add(key)
    return dict(grouped)


def attach_gold_chunks(
    eval_questions: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunks_by_document: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_document[chunk["document_id"]].append(chunk["chunk_id"])

    qrel_docs_by_query = qrels_by_query(qrels)
    eval_rows = []
    questions_without_qrels = []
    questions_without_gold_chunks = []
    missing_qrel_document_ids = []

    for question in eval_questions:
        query_id = question.get("query_id", "")
        document_ids = qrel_docs_by_query.get(query_id, [])
        if not document_ids:
            questions_without_qrels.append(query_id)
            document_ids = list(question.get("ground_truth_document_ids") or [])

        gold_chunks_by_document = {
            document_id: chunks_by_document.get(document_id, []) for document_id in document_ids
        }
        for document_id, chunk_ids in gold_chunks_by_document.items():
            if not chunk_ids:
                missing_qrel_document_ids.append(document_id)

        gold_chunk_ids = [
            chunk_id
            for document_id in document_ids
            for chunk_id in gold_chunks_by_document.get(document_id, [])
        ]
        if not gold_chunk_ids:
            questions_without_gold_chunks.append(query_id)

        eval_row = {
            **question,
            "ground_truth_document_ids": document_ids,
            "gold_chunk_ids": gold_chunk_ids,
            "gold_chunks_by_document": gold_chunks_by_document,
            "gold_document_count": len(document_ids),
            "gold_chunk_count": len(gold_chunk_ids),
            "gold_chunk_mapping_version": "document_qrels_all_chunks_v1",
        }
        eval_rows.append(eval_row)

    diagnostics = {
        "questions_without_qrels": questions_without_qrels,
        "questions_without_gold_chunks": questions_without_gold_chunks,
        "missing_qrel_document_ids": sorted(set(missing_qrel_document_ids)),
        "qrel_query_ids_without_eval_question": sorted(
            set(qrel_docs_by_query) - {row.get("query_id", "") for row in eval_questions}
        ),
    }
    return eval_rows, diagnostics


def make_quality_rows(
    documents: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    eval_questions_with_gold: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    text_field: str,
) -> list[dict[str, Any]]:
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    document_ids = [doc.get("document_id", "") for doc in documents]
    text_lengths = [len(doc.get(text_field) or "") for doc in documents]
    chunk_lengths = [chunk["char_length"] for chunk in chunks]
    chunk_counts_by_doc = Counter(chunk["document_id"] for chunk in chunks)
    gold_chunk_counts = [row["gold_chunk_count"] for row in eval_questions_with_gold]

    rows = [
        {"metric": "document_count", "value": len(documents)},
        {"metric": "qrel_count", "value": len(qrels)},
        {"metric": "eval_question_count", "value": len(eval_questions_with_gold)},
        {"metric": "chunk_count", "value": len(chunks)},
        {"metric": "duplicate_document_id_count", "value": len(document_ids) - len(set(document_ids))},
        {"metric": "duplicate_chunk_id_count", "value": len(chunk_ids) - len(set(chunk_ids))},
        {"metric": "blank_document_id_count", "value": sum(1 for value in document_ids if not value)},
        {"metric": f"blank_{text_field}_count", "value": sum(1 for value in text_lengths if value == 0)},
        {"metric": f"{text_field}_min_length", "value": min(text_lengths) if text_lengths else 0},
        {"metric": f"{text_field}_avg_length", "value": round(mean(text_lengths), 3) if text_lengths else 0},
        {"metric": f"{text_field}_max_length", "value": max(text_lengths) if text_lengths else 0},
        {"metric": "chunk_min_length", "value": min(chunk_lengths) if chunk_lengths else 0},
        {"metric": "chunk_avg_length", "value": round(mean(chunk_lengths), 3) if chunk_lengths else 0},
        {"metric": "chunk_max_length", "value": max(chunk_lengths) if chunk_lengths else 0},
        {"metric": "chunks_per_document_min", "value": min(chunk_counts_by_doc.values()) if chunks else 0},
        {"metric": "chunks_per_document_avg", "value": round(mean(chunk_counts_by_doc.values()), 3) if chunks else 0},
        {"metric": "chunks_per_document_max", "value": max(chunk_counts_by_doc.values()) if chunks else 0},
        {"metric": "gold_chunk_count_min", "value": min(gold_chunk_counts) if gold_chunk_counts else 0},
        {"metric": "gold_chunk_count_avg", "value": round(mean(gold_chunk_counts), 3) if gold_chunk_counts else 0},
        {"metric": "gold_chunk_count_max", "value": max(gold_chunk_counts) if gold_chunk_counts else 0},
        {"metric": "questions_without_qrels_count", "value": len(diagnostics["questions_without_qrels"])},
        {
            "metric": "questions_without_gold_chunks_count",
            "value": len(diagnostics["questions_without_gold_chunks"]),
        },
        {"metric": "missing_qrel_document_id_count", "value": len(diagnostics["missing_qrel_document_ids"])},
        {
            "metric": "qrel_query_ids_without_eval_question_count",
            "value": len(diagnostics["qrel_query_ids_without_eval_question"]),
        },
    ]
    return rows


def make_manifest(
    *,
    version: str,
    documents_path: Path,
    eval_questions_path: Path,
    qrels_path: Path,
    output_dir: Path,
    text_field: str,
    chunk_size: int,
    chunk_overlap: int,
    documents: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    eval_questions_with_gold: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    text_lengths = [len(doc.get(text_field) or "") for doc in documents]
    chunk_lengths = [chunk["char_length"] for chunk in chunks]
    chunk_counts_by_doc = Counter(chunk["document_id"] for chunk in chunks)
    gold_chunk_counts = [row["gold_chunk_count"] for row in eval_questions_with_gold]
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    return {
        "version": version,
        "created_at": datetime.now(KST).isoformat(timespec="seconds"),
        "input": {
            "documents_path": str(documents_path),
            "eval_questions_path": str(eval_questions_path),
            "qrels_path": str(qrels_path),
        },
        "output": {
            "output_dir": str(output_dir),
            "chunks_path": str(output_dir / "chunks.jsonl"),
            "eval_questions_with_gold_chunks_path": str(
                output_dir / "eval_questions_with_gold_chunks.jsonl"
            ),
            "chunk_quality_report_path": str(output_dir / "chunk_quality_report.csv"),
            "chunk_manifest_path": str(output_dir / "chunk_manifest.json"),
        },
        "chunking": {
            "text_field": text_field,
            "unit": "character",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "step": chunk_size - chunk_overlap,
            "chunk_id_pattern": "{document_id}__chunk_{chunk_index:03d}",
        },
        "counts": {
            "document_count": len(documents),
            "qrel_count": len(qrels),
            "eval_question_count": len(eval_questions_with_gold),
            "chunk_count": len(chunks),
            "unique_chunk_id_count": len(set(chunk_ids)),
        },
        "retrieval_text_stats": {
            "min_length": min(text_lengths) if text_lengths else 0,
            "avg_length": round(mean(text_lengths), 3) if text_lengths else 0,
            "p50_length": percentile(text_lengths, 50),
            "p75_length": percentile(text_lengths, 75),
            "p90_length": percentile(text_lengths, 90),
            "p95_length": percentile(text_lengths, 95),
            "max_length": max(text_lengths) if text_lengths else 0,
        },
        "chunk_stats": {
            "min_length": min(chunk_lengths) if chunk_lengths else 0,
            "avg_length": round(mean(chunk_lengths), 3) if chunk_lengths else 0,
            "p50_length": percentile(chunk_lengths, 50),
            "p90_length": percentile(chunk_lengths, 90),
            "max_length": max(chunk_lengths) if chunk_lengths else 0,
            "chunks_per_document_min": min(chunk_counts_by_doc.values()) if chunks else 0,
            "chunks_per_document_avg": round(mean(chunk_counts_by_doc.values()), 3) if chunks else 0,
            "chunks_per_document_max": max(chunk_counts_by_doc.values()) if chunks else 0,
        },
        "gold_chunk_mapping": {
            "strategy": "For each positive qrel document_id, mark every chunk from that document as a gold chunk candidate.",
            "gold_chunk_count_min": min(gold_chunk_counts) if gold_chunk_counts else 0,
            "gold_chunk_count_avg": round(mean(gold_chunk_counts), 3) if gold_chunk_counts else 0,
            "gold_chunk_count_max": max(gold_chunk_counts) if gold_chunk_counts else 0,
        },
        "validation": {
            "duplicate_chunk_id_count": len(chunk_ids) - len(set(chunk_ids)),
            "questions_without_qrels_count": len(diagnostics["questions_without_qrels"]),
            "questions_without_gold_chunks_count": len(diagnostics["questions_without_gold_chunks"]),
            "missing_qrel_document_id_count": len(diagnostics["missing_qrel_document_ids"]),
            "qrel_query_ids_without_eval_question_count": len(
                diagnostics["qrel_query_ids_without_eval_question"]
            ),
            "diagnostics": diagnostics,
        },
    }


def run(
    *,
    version: str,
    documents_path: Path,
    eval_questions_path: Path,
    qrels_path: Path,
    output_dir: Path,
    text_field: str,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, Any]:
    documents = read_jsonl(documents_path)
    eval_questions = read_jsonl(eval_questions_path)
    qrels = read_jsonl(qrels_path)

    chunks = build_chunks(
        documents,
        version=version,
        text_field=text_field,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    eval_questions_with_gold, diagnostics = attach_gold_chunks(eval_questions, qrels, chunks)
    quality_rows = make_quality_rows(
        documents,
        qrels,
        chunks,
        eval_questions_with_gold,
        diagnostics,
        text_field,
    )
    manifest = make_manifest(
        documents_path=documents_path,
        version=version,
        eval_questions_path=eval_questions_path,
        qrels_path=qrels_path,
        output_dir=output_dir,
        text_field=text_field,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        documents=documents,
        qrels=qrels,
        chunks=chunks,
        eval_questions_with_gold=eval_questions_with_gold,
        diagnostics=diagnostics,
    )

    write_jsonl(output_dir / "chunks.jsonl", chunks)
    write_jsonl(output_dir / "eval_questions_with_gold_chunks.jsonl", eval_questions_with_gold)
    write_csv(output_dir / "chunk_quality_report.csv", quality_rows, ["metric", "value"])
    write_json(output_dir / "chunk_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Ontong Youth chunking v1 artifacts.")
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--eval-questions", type=Path, default=DEFAULT_EVAL_QUESTIONS)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--text-field", default="retrieval_text")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run(
        version=args.version,
        documents_path=args.documents,
        eval_questions_path=args.eval_questions,
        qrels_path=args.qrels,
        output_dir=args.output_dir,
        text_field=args.text_field,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(json.dumps(manifest["validation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data/processed/ontong_youth_chunks_v1/chunks.jsonl"
DEFAULT_EVAL = ROOT / "data/processed/ontong_youth_chunks_v1/eval_questions_with_gold_chunks.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data/processed/ontong_youth_chunks_v1/evaluation"
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


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[가-힣A-Za-z0-9]+", (text or "").lower())
    tokens = words[:]
    compact = "".join(words)
    for n in (2, 3):
        tokens.extend(compact[i : i + n] for i in range(max(0, len(compact) - n + 1)))
    return [token for token in tokens if token]


class SimpleBM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunk_ids: list[str] = []
        self.doc_lens: list[int] = []
        self.avgdl = 0.0
        self.idf: dict[str, float] = {}
        self.postings: dict[str, list[tuple[int, int]]] = {}

    def fit(self, chunk_ids: list[str], texts: list[str]) -> "SimpleBM25":
        self.chunk_ids = chunk_ids
        dfs: Counter[str] = Counter()
        postings: dict[str, list[tuple[int, int]]] = {}
        for index, text in enumerate(texts):
            counts = Counter(tokenize(text))
            self.doc_lens.append(sum(counts.values()))
            dfs.update(counts.keys())
            for term, tf in counts.items():
                postings.setdefault(term, []).append((index, tf))
        n_docs = len(chunk_ids)
        self.avgdl = mean(self.doc_lens) if self.doc_lens else 0.0
        self.idf = {
            term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in dfs.items()
        }
        self.postings = postings
        return self

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        scores: dict[int, float] = {}
        for term in tokenize(query):
            idf = self.idf.get(term, 0.0)
            if not idf:
                continue
            for index, tf in self.postings.get(term, []):
                dl = self.doc_lens[index] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[index] = scores.get(index, 0.0) + idf * (tf * (self.k1 + 1)) / denom
        top_indexes = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [
            {
                "chunk_id": self.chunk_ids[index],
                "rank": rank,
                "score": float(scores[index]),
            }
            for rank, index in enumerate(top_indexes, start=1)
            if scores[index] > 0
        ]


class CharTfidfRetriever:
    def __init__(self, max_features: int = 80000) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=max_features,
            lowercase=True,
        )
        self.chunk_ids: list[str] = []
        self.matrix = None

    def fit(self, chunk_ids: list[str], texts: list[str]) -> "CharTfidfRetriever":
        self.chunk_ids = chunk_ids
        self.matrix = normalize(self.vectorizer.fit_transform(texts), norm="l2", axis=1)
        return self

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self.matrix is None:
            return []
        query_vector = normalize(self.vectorizer.transform([query]), norm="l2", axis=1)
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        top_indexes = np.argsort(-scores)[:top_k]
        return [
            {
                "chunk_id": self.chunk_ids[int(index)],
                "rank": rank,
                "score": float(scores[int(index)]),
            }
            for rank, index in enumerate(top_indexes, start=1)
            if scores[int(index)] > 0
        ]


def rrf(result_sets: list[list[dict[str, Any]]], weights: list[float], k: int = 60) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for results, weight in zip(result_sets, weights):
        for rank, row in enumerate(results, start=1):
            chunk_id = row["chunk_id"]
            item = fused.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "score": 0.0,
                    "rank": None,
                },
            )
            item["score"] += weight / (k + rank)
    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


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


def evaluate_ranked(
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


def compact_top_results(
    results: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    max_items: int = 5,
) -> list[dict[str, Any]]:
    compact = []
    for row in results[:max_items]:
        chunk = chunks_by_id[row["chunk_id"]]
        compact.append(
            {
                "rank": row["rank"],
                "chunk_id": row["chunk_id"],
                "document_id": chunk["document_id"],
                "score": round(float(row["score"]), 6),
                "title": chunk.get("title", ""),
                "text_preview": " ".join(chunk.get("text", "").split())[:220],
            }
        )
    return compact


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = sorted({row["method"] for row in rows})
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
    summary = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        summary.append(
            {
                "method": method,
                "query_count": len(method_rows),
                **{
                    metric: round(mean(float(row[metric]) for row in method_rows), 6)
                    for metric in metric_names
                },
            }
        )
    return summary


def make_report(summary: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    lines = [
        "# Ontong Youth Chunk Retrieval Evaluation v1",
        "",
        f"- created_at: {manifest['created_at']}",
        f"- chunk_count: {manifest['counts']['chunk_count']}",
        f"- eval_question_count: {manifest['counts']['eval_question_count']}",
        f"- gold mapping: {manifest['gold_chunk_mapping']['strategy']}",
        "",
        "| method | chunk Hit@1 | chunk Hit@3 | chunk Hit@5 | chunk MRR | document Hit@1 | document Hit@5 | document MRR | latency ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| {method} | {chunk_hit@1:.4f} | {chunk_hit@3:.4f} | {chunk_hit@5:.4f} | {chunk_mrr:.4f} | "
            "{document_hit@1:.4f} | {document_hit@5:.4f} | {document_mrr:.4f} | {latency_ms:.2f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- chunk Hit@k checks whether any retrieved chunk_id is in gold_chunk_ids.",
            "- document Hit@k checks whether any retrieved chunk's document_id is in ground_truth_document_ids.",
            "- v1 gold_chunk_ids are broad candidates: every chunk from a qrel-positive document is counted as gold.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(
    *,
    chunks_path: Path,
    eval_path: Path,
    output_dir: Path,
    top_k: int,
    max_features: int,
) -> dict[str, Any]:
    chunks = read_jsonl(chunks_path)
    eval_questions = read_jsonl(eval_path)
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    texts = [chunk.get("text", "") for chunk in chunks]
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    chunk_to_document = {chunk["chunk_id"]: chunk["document_id"] for chunk in chunks}

    build_start = time.perf_counter()
    bm25 = SimpleBM25().fit(chunk_ids, texts)
    dense = CharTfidfRetriever(max_features=max_features).fit(chunk_ids, texts)
    build_elapsed_ms = (time.perf_counter() - build_start) * 1000

    methods = {
        "BM25": lambda query: bm25.search(query, top_k),
        "Char TF-IDF": lambda query: dense.search(query, top_k),
        "Hybrid RRF": lambda query: rrf(
            [bm25.search(query, top_k=max(top_k, 50)), dense.search(query, top_k=max(top_k, 50))],
            weights=[1.0, 1.0],
        )[:top_k],
    }

    metric_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for method, search in methods.items():
        for question in eval_questions:
            query = question["query"]
            started = time.perf_counter()
            results = search(query)
            elapsed_ms = (time.perf_counter() - started) * 1000
            ranked_chunk_ids = [row["chunk_id"] for row in results]
            gold_chunk_ids = set(question.get("gold_chunk_ids") or [])
            gold_document_ids = set(question.get("ground_truth_document_ids") or [])
            metrics = evaluate_ranked(
                ranked_chunk_ids,
                gold_chunk_ids,
                chunk_to_document,
                gold_document_ids,
            )
            metric_row = {
                "method": method,
                "query_id": question["query_id"],
                "question_type": question.get("question_type", ""),
                "primary_category": question.get("primary_category", ""),
                "primary_mclsf": question.get("primary_mclsf", ""),
                "gold_document_count": len(gold_document_ids),
                "gold_chunk_count": len(gold_chunk_ids),
                "retrieved_count": len(ranked_chunk_ids),
                "latency_ms": elapsed_ms,
                **metrics,
            }
            metric_rows.append(metric_row)
            result_row = {
                "method": method,
                "query_id": question["query_id"],
                "query": query,
                "ground_truth_document_ids": list(gold_document_ids),
                "gold_chunk_ids": list(gold_chunk_ids),
                "top_results": compact_top_results(results, chunks_by_id, max_items=10),
                **metrics,
            }
            result_rows.append(result_row)
            if metrics["document_hit@5"] == 0:
                failure_rows.append(result_row)

    summary = summarize(metric_rows)
    manifest = {
        "version": "ontong_youth_chunk_retrieval_eval_v1",
        "created_at": datetime.now(KST).isoformat(timespec="seconds"),
        "input": {
            "chunks_path": str(chunks_path),
            "eval_questions_with_gold_chunks_path": str(eval_path),
        },
        "output": {
            "output_dir": str(output_dir),
            "metrics_path": str(output_dir / "chunk_retrieval_metrics.csv"),
            "summary_path": str(output_dir / "chunk_retrieval_summary.csv"),
            "top_results_path": str(output_dir / "chunk_retrieval_top_results.jsonl"),
            "failure_cases_path": str(output_dir / "chunk_retrieval_failure_cases.jsonl"),
            "report_path": str(output_dir / "chunk_retrieval_report.md"),
        },
        "retrieval": {
            "top_k": top_k,
            "methods": list(methods),
            "dense_backend": "sklearn_char_wb_tfidf",
            "bm25_tokenizer": "korean_alnum_words_plus_2_3_char_ngrams",
            "max_features": max_features,
            "build_elapsed_ms": round(build_elapsed_ms, 3),
        },
        "counts": {
            "chunk_count": len(chunks),
            "eval_question_count": len(eval_questions),
            "metric_row_count": len(metric_rows),
            "failure_case_count": len(failure_rows),
        },
        "gold_chunk_mapping": {
            "strategy": "v1 broad candidate mapping: all chunks from each qrel-positive document are gold.",
        },
        "summary": summary,
    }

    write_csv(output_dir / "chunk_retrieval_metrics.csv", metric_rows)
    write_csv(output_dir / "chunk_retrieval_summary.csv", summary)
    write_jsonl(output_dir / "chunk_retrieval_top_results.jsonl", result_rows)
    write_jsonl(output_dir / "chunk_retrieval_failure_cases.jsonl", failure_rows)
    write_json(output_dir / "chunk_retrieval_eval_manifest.json", manifest)
    (output_dir / "chunk_retrieval_report.md").write_text(
        make_report(summary, manifest),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Ontong Youth chunk retrieval v1.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-features", type=int, default=80000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run(
        chunks_path=args.chunks,
        eval_path=args.eval,
        output_dir=args.output_dir,
        top_k=args.top_k,
        max_features=args.max_features,
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

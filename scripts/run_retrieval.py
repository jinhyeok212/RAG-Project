#!/usr/bin/env python
"""Run read-only Top-k retrieval for a JSONL question set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.embedding import load_existing_for_search  # noqa: E402
from src.rag.retriever import RetrievalConfig, Retriever  # noqa: E402

DEFAULT_QUESTIONS = PROJECT_ROOT / "data" / "processed" / "ontong_youth_eval_questions_v2_final_47" / "eval_questions.jsonl"
DEFAULT_DB_DIR = PROJECT_ROOT / "indexes" / "chroma" / "chroma_db"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "exp_baseline_kure_v1" / "retrieval_results.jsonl"
DEFAULT_COLLECTION = "c2_kure_v1_section800_v1"
DEFAULT_MODEL = "nlpai-lab/KURE-v1"


def read_questions(path: Path) -> Iterator[tuple[str, str]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            query_id, query = row.get("query_id"), row.get("query")
            if not isinstance(query_id, str) or not isinstance(query, str):
                raise ValueError(f"query_id and query must be strings at {path}:{line_number}")
            yield query_id, query


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if not args.questions.is_file():
        raise FileNotFoundError(f"Question file not found: {args.questions}")

    # Uses PersistentClient.get_collection(); never creates or mutates the DB.
    embedder, collection = load_existing_for_search(
        db_dir=args.db_dir,
        collection_name=args.collection,
        model_name=args.model,
        device=args.device,
        embed_batch_size=1,
    )
    retriever = Retriever(
        embedder,
        collection,
        RetrievalConfig(embedding_model=args.model, evaluation_max_k=args.top_k),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for query_id, query in read_questions(args.questions):
            stream.write(json.dumps(retriever.retrieve(query_id, query), ensure_ascii=False) + "\n")
            count += 1
    print(f"Retrieved {count} questions")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()

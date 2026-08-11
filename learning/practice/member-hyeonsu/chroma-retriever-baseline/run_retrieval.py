"""질문으로 Top-k 검색을 실행하고 결과를 JSONL로 저장하는 실제 Retriever 진입점."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import load_settings
from src.embedding import TextEmbedder
from src.retriever import Retriever
from src.vector_store import ChromaStore


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """한 검색 결과를 한 줄에 하나씩 기록해 후속 평가가 쉽게 한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(config_path: str, cli_top_k: list[int] | None) -> None:
    settings = load_settings(config_path)
    top_k_values = cli_top_k or settings.top_k_values
    if any(k <= 0 for k in top_k_values):
        raise ValueError("Top-k는 1 이상의 정수여야 합니다.")

    questions = read_jsonl(settings.questions_path)
    embedder = TextEmbedder(settings.embedding_model)
    retriever = Retriever(
        embedder=embedder,
        store=ChromaStore(settings),
        distance_metric=settings.distance_metric,
        chunk_id_field=settings.chunk_id_field,
        document_id_field=settings.document_id_field,
    )

    # k마다 실제 DB query를 다시 실행하므로 Top-1/3/5 각각의 검색 시간이 측정된다.
    for top_k in top_k_values:
        all_results: list[dict[str, Any]] = []
        for item in questions:
            all_results.extend(
                retriever.retrieve(item["question_id"], item["question"], top_k)
            )

        output_path = settings.output_dir / f"retrieval_top_{top_k}.jsonl"
        write_jsonl(output_path, all_results)
        print(f"Top-{top_k}: {len(all_results)}개 결과 저장 -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chroma Top-k 검색 실험")
    parser.add_argument("--config", default="config.json", help="설정 JSON 경로")
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        help="설정값 대신 사용할 k 목록 (예: --top-k 1 3 5 10)",
    )
    args = parser.parse_args()
    main(args.config, args.top_k)

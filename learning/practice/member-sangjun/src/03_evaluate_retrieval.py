"""질문으로 검색하고 answer 포함 여부를 평가하는 단계."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_utils import (
    OUTPUT_DIR,
    SAMPLE_DIR,
    VECTOR_DIR,
    answer_in_text,
    e5_query,
    ensure_project_dirs,
    read_faiss_index,
    read_csv,
    read_jsonl,
    write_csv,
)


def import_search_dependencies():
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "필요한 패키지가 없습니다. 먼저 다음 명령을 실행하세요:\n"
            "pip install -r requirements.txt"
        ) from exc
    return faiss, np, SentenceTransformer


def load_index_config(index_dir: Path) -> dict:
    config_path = index_dir / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def evaluate(index_dir: Path, eval_path: Path, top_k: int, model_name: str | None) -> dict:
    faiss, np, SentenceTransformer = import_search_dependencies()

    config = load_index_config(index_dir)
    model_name = model_name or config.get("model_name", "intfloat/multilingual-e5-small")

    index = read_faiss_index(faiss, index_dir / "index.faiss")
    metadata = read_jsonl(index_dir / "metadata.jsonl")
    eval_rows = read_csv(eval_path)

    model = SentenceTransformer(model_name)
    questions = [e5_query(row["question"]) for row in eval_rows]
    question_embeddings = model.encode(
        questions,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(np.asarray(question_embeddings, dtype="float32"), top_k)

    detail_rows = []
    found_count = 0

    for row_idx, eval_row in enumerate(eval_rows):
        answer = eval_row["answer"]
        found_rank = ""
        combined_texts = []

        for rank, (score, chunk_index) in enumerate(zip(scores[row_idx], indices[row_idx]), start=1):
            if chunk_index < 0:
                continue

            chunk = metadata[int(chunk_index)]
            combined_texts.append(chunk["text"])
            is_answer_in_chunk = answer_in_text(answer, chunk["text"])
            if is_answer_in_chunk and not found_rank:
                found_rank = str(rank)

            detail_rows.append(
                {
                    "question_id": eval_row["question_id"],
                    "question": eval_row["question"],
                    "answer": answer,
                    "expected_doc_id": eval_row["doc_id"],
                    "rank": rank,
                    "score": round(float(score), 6),
                    "retrieved_doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "answer_in_chunk": "O" if is_answer_in_chunk else "X",
                    "retrieved_text": chunk["text"],
                }
            )

        if found_rank or answer_in_text(answer, " ".join(combined_texts)):
            found_count += 1

    question_count = len(eval_rows)
    recall = found_count / question_count if question_count else 0.0

    chunk_size = config.get("chunk_size", "unknown")
    detail_path = OUTPUT_DIR / f"retrieval_results_chunk_{chunk_size}_top_{top_k}.csv"
    summary_path = OUTPUT_DIR / f"eval_summary_chunk_{chunk_size}_top_{top_k}.csv"

    write_csv(
        detail_path,
        detail_rows,
        [
            "question_id",
            "question",
            "answer",
            "expected_doc_id",
            "rank",
            "score",
            "retrieved_doc_id",
            "chunk_id",
            "answer_in_chunk",
            "retrieved_text",
        ],
    )
    write_csv(
        summary_path,
        [
            {
                "model_name": model_name,
                "chunk_size": chunk_size,
                "top_k": top_k,
                "question_count": question_count,
                "answer_found_count": found_count,
                "recall": round(recall, 4),
            }
        ],
        ["model_name", "chunk_size", "top_k", "question_count", "answer_found_count", "recall"],
    )

    return {
        "question_count": question_count,
        "answer_found_count": found_count,
        "recall": recall,
        "detail_path": detail_path,
        "summary_path": summary_path,
    }


def main():
    parser = argparse.ArgumentParser(description="질문으로 FAISS 검색을 실행하고 정답 포함 여부를 평가합니다.")
    parser.add_argument("--index-dir", type=Path, default=VECTOR_DIR / "faiss" / "chunk_300")
    parser.add_argument("--eval", type=Path, default=SAMPLE_DIR / "eval_questions.csv")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    ensure_project_dirs()
    result = evaluate(args.index_dir, args.eval, args.top_k, args.model_name)

    print(f"질문 수: {result['question_count']}")
    print(f"정답 포함 수: {result['answer_found_count']}")
    print(f"Recall: {result['recall']:.2%}")
    print(f"상세 결과 저장: {result['detail_path']}")
    print(f"요약 결과 저장: {result['summary_path']}")


if __name__ == "__main__":
    main()

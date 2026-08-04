"""AIHub 원본 데이터에서 소량 샘플을 만드는 단계."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_utils import (
    SAMPLE_DIR,
    default_normal_data_path,
    ensure_project_dirs,
    load_json_from_path,
    normalize_whitespace,
    write_csv,
    write_jsonl,
)


def build_sample(input_path: Path, max_contexts: int, max_questions: int) -> tuple[int, int]:
    """AIHub Normal 데이터에서 context 문서와 평가 질문을 추출한다."""
    raw = load_json_from_path(input_path)
    data = raw["data"]

    documents = []
    eval_rows = []
    seen_contexts: set[str] = set()

    for article_idx, article in enumerate(data):
        title = str(article.get("title", ""))
        source = str(article.get("source", ""))
        paragraphs = article.get("paragraphs", [])

        for paragraph_idx, paragraph in enumerate(paragraphs):
            context = normalize_whitespace(paragraph.get("context", ""))
            if not context or context in seen_contexts:
                continue

            doc_id = f"doc_{len(documents) + 1:04d}"
            seen_contexts.add(context)
            documents.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "source": source,
                    "article_idx": article_idx,
                    "paragraph_idx": paragraph_idx,
                    "text": context,
                }
            )

            for qa in paragraph.get("qas", []):
                if len(eval_rows) >= max_questions:
                    break

                answers = qa.get("answers", [])
                if not answers:
                    continue

                answer_text = normalize_whitespace(answers[0].get("text", ""))
                question = normalize_whitespace(qa.get("question", ""))
                if not question or not answer_text:
                    continue

                eval_rows.append(
                    {
                        "question_id": qa.get("id", f"q_{len(eval_rows) + 1:04d}"),
                        "doc_id": doc_id,
                        "question": question,
                        "answer": answer_text,
                        "title": title,
                    }
                )

            if len(documents) >= max_contexts and len(eval_rows) >= max_questions:
                break

        if len(documents) >= max_contexts and len(eval_rows) >= max_questions:
            break

    write_jsonl(SAMPLE_DIR / "documents.jsonl", documents)
    write_csv(
        SAMPLE_DIR / "eval_questions.csv",
        eval_rows,
        ["question_id", "doc_id", "question", "answer", "title"],
    )

    return len(documents), len(eval_rows)


def main():
    parser = argparse.ArgumentParser(description="AIHub 기계독해 데이터에서 소량 샘플을 만듭니다.")
    parser.add_argument("--input", type=Path, default=default_normal_data_path())
    parser.add_argument("--max-contexts", type=int, default=200)
    parser.add_argument("--max-questions", type=int, default=30)
    args = parser.parse_args()

    ensure_project_dirs()
    doc_count, question_count = build_sample(args.input, args.max_contexts, args.max_questions)

    print(f"문서 샘플 저장: {SAMPLE_DIR / 'documents.jsonl'} ({doc_count}개)")
    print(f"평가 질문 저장: {SAMPLE_DIR / 'eval_questions.csv'} ({question_count}개)")


if __name__ == "__main__":
    main()

# KorQuAD 다운로드
# → 일부 질문 선택
# → context 중복 제거
# → documents.json 저장
# → questions.json 저장

import hashlib # 암호화 해시 함수를 사용하여 동일한 context에 대해 동일한 문서 ID를 생성하기 위해 hashlib 모듈을 임포트
import json
from pathlib import Path

from datasets import load_dataset


# 현재 파일: korquad-rag/src/load_data.py
# parents[1]: korquad-rag/
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

DOCUMENTS_PATH = DATA_DIR / "documents.json"
QUESTIONS_PATH = DATA_DIR / "questions.json"


def create_document_id(context: str) -> str:
    """
    동일한 context에는 동일한 문서 ID가 생성되도록
    MD5 해시를 사용한다.
    """
    return hashlib.md5(context.encode("utf-8")).hexdigest()


def prepare_korquad_data(
    question_count: int = 100,
    split: str = "train"
) -> tuple[list[dict], list[dict]]:
    """
    KorQuAD를 다운로드하고,
    RAG 검색용 문서와 평가용 질문을 분리한다.
    """

    print("KorQuAD 데이터셋을 불러오는 중입니다.")

    dataset = load_dataset(
        "KorQuAD/squad_kor_v1",
        split=split
    )

    # 처음에는 전체가 아닌 일부 질문만 사용한다.
    selected_data = dataset.select(
        range(min(question_count, len(dataset)))
    )

    documents_by_id: dict[str, dict] = {}
    questions: list[dict] = []

    for row in selected_data:
        context = row["context"].strip()
        doc_id = create_document_id(context)

        # 동일한 context가 여러 질문에 반복되므로 중복 제거
        if doc_id not in documents_by_id:
            documents_by_id[doc_id] = {
                "doc_id": doc_id,
                "title": row["title"],
                "text": context
            }

        answer_texts = row["answers"].get("text", [])
        answer_starts = row["answers"].get("answer_start", [])

        ground_truth_answer = (
            answer_texts[0] if answer_texts else ""
        )

        answer_start = (
            answer_starts[0] if answer_starts else -1
        )

        questions.append({
            "question_id": row["id"],
            "question": row["question"],
            "ground_truth_answer": ground_truth_answer,
            "answer_start": answer_start,
            "ground_truth_doc_id": doc_id
        })

    documents = list(documents_by_id.values())

    return documents, questions


def save_json(data: list[dict], file_path: Path) -> None:
    """
    Python 데이터를 JSON 파일로 저장한다.
    """
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with file_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )


def main() -> None:
    documents, questions = prepare_korquad_data(
        question_count=100,
        split="train"
    )

    save_json(documents, DOCUMENTS_PATH)
    save_json(questions, QUESTIONS_PATH)

    print(f"고유 문서 수: {len(documents)}")
    print(f"질문 수: {len(questions)}")
    print(f"문서 저장 위치: {DOCUMENTS_PATH}")
    print(f"질문 저장 위치: {QUESTIONS_PATH}")


if __name__ == "__main__":
    main()
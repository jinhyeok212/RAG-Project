"""Training 질문 5,000개의 Chroma 인덱스를 생성합니다."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.chroma_store import ChromaQuestionStore
from src.embedding_model import KoreanEmbeddingModel
from src.retrieval_utils import ensure_unique, env_int, load_environment, read_and_validate_csv


def main() -> None:
    """CSV 검증, 임베딩, upsert, 저장 개수 검증을 순서대로 실행합니다."""
    load_environment()
    csv_path = PROJECT_ROOT / "data" / "experiment" / "super_train_sample_5000.csv"
    required = [
        "document_id", "category", "conversation_id", "qa_number",
        "intent", "question", "answer",
    ]
    frame = read_and_validate_csv(csv_path, required)
    if len(frame) != 5_000:
        raise ValueError(f"Training 표본은 5,000개여야 합니다. 실제: {len(frame)}")
    ensure_unique(frame["document_id"], "document_id")

    batch_size = env_int("EMBEDDING_BATCH_SIZE", 64)
    model = KoreanEmbeddingModel()
    questions = frame["question"].tolist()
    started = time.perf_counter()
    embeddings = model.encode(questions, batch_size=batch_size)
    elapsed = time.perf_counter() - started

    metadatas = [
        {
            "answer": row.answer,
            "category": row.category,
            "intent": row.intent,
            "conversation_id": row.conversation_id,
            "qa_number": row.qa_number,
        }
        for row in frame.itertuples(index=False)
    ]
    store = ChromaQuestionStore()
    store.upsert(frame["document_id"].tolist(), questions, embeddings, metadatas)
    if store.count != len(frame):
        raise RuntimeError(
            f"Chroma 문서 수 검증 실패: 기대 {len(frame)}, 실제 {store.count}. "
            "기존 컬렉션 설정과 내용을 확인하세요."
        )

    print(f"모델명: {model.model_name}")
    print(f"임베딩 차원: {model.dimension}")
    print(f"임베딩한 문서 수: {len(frame)}")
    print(f"전체 임베딩 시간: {elapsed:.3f}초")
    print(f"평균 문서 임베딩 시간: {elapsed * 1000 / len(frame):.3f}ms")
    print(f"Chroma 경로: {store.path}")
    print(f"컬렉션: {store.collection_name}")
    print(f"실제 저장 문서 수: {store.count} (검증 성공)")


if __name__ == "__main__":
    main()

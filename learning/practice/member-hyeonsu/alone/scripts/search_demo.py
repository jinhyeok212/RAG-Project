"""한국어 질문을 입력받아 상위 검색 결과를 보여주는 데모."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.chroma_store import ChromaQuestionStore
from src.embedding_model import KoreanEmbeddingModel
from src.retrieval_utils import env_int, load_environment


def main() -> None:
    """빈 입력이나 빈 컬렉션에도 오류 없이 종료합니다."""
    load_environment()
    top_k = env_int("TOP_K", 5)
    store = ChromaQuestionStore()
    if store.count == 0:
        print("검색할 문서가 없습니다. 먼저 build_chroma_index.py를 실행하세요.")
        return
    query = input("한국어 질문을 입력하세요: ").strip()
    if not query:
        print("질문이 비어 있어 검색하지 않습니다.")
        return
    model = KoreanEmbeddingModel()
    embedding = model.encode([query])[0]
    results = store.query(embedding, top_k)
    if not results:
        print("검색 결과가 없습니다.")
        return
    for result in results:
        metadata = result["metadata"]
        print("\n" + "=" * 72)
        print(f"검색 순위: {result['rank']}")
        print(f"거리(낮을수록 가까움): {result['distance']:.6f}")
        print(f"document_id: {result['document_id']}")
        print(f"기존 고객 질문: {result['question']}")
        print(f"상담 답변: {metadata.get('answer', '')}")
        print(f"인텐트: {metadata.get('intent', '')}")
        print(f"카테고리: {metadata.get('category', '')}")


if __name__ == "__main__":
    main()

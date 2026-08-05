"""터미널에서 반복 실행하는 최소 RAG 대화 데모."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_service import RAGService


def _print_result(result: dict) -> None:
    print(f"\n[최종 답변]\n{result['final_answer']}")
    print("\n[검색 근거]")
    documents = result["retrieved_documents"]
    if not documents:
        print("검색된 문서가 없습니다.")
    for document in documents:
        print(f"{document['rank']}. 기존 질문: {document['retrieved_question']}")
        print(f"   기존 답변: {document['retrieved_answer']}")
        print(f"   인텐트: {document['intent']}")
        print(f"   카테고리: {document['category']}")
        print(f"   문서 ID: {document['document_id']}")
        distance = document.get("distance")
        distance_text = f"{distance:.6f}" if isinstance(distance, (int, float)) else "N/A"
        print(f"   거리(낮을수록 가까움): {distance_text}\n")
    print("[처리 시간]")
    print(f"질문 임베딩: {result['query_embedding_time_ms']:.2f} ms")
    print(f"검색: {result['retrieval_time_ms']:.2f} ms")
    print(f"LLM 생성: {result['generation_time_ms']:.2f} ms")
    print(f"전체: {result['total_time_ms']:.2f} ms")


def main() -> None:
    try:
        service = RAGService()
    except Exception as exc:
        print(f"RAG 데모를 시작할 수 없습니다: {exc}")
        return
    print("RAG 상담 데모입니다. 종료하려면 exit, quit 또는 종료를 입력하세요.")
    while True:
        try:
            query = input("\n질문을 입력하세요: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n데모를 종료합니다.")
            break
        if query.lower() in {"exit", "quit"} or query == "종료":
            print("데모를 종료합니다.")
            break
        if not query:
            print("질문이 비어 있습니다. 질문을 입력해주세요.")
            continue
        _print_result(service.ask(query))


if __name__ == "__main__":
    main()

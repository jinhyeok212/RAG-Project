"""검색 근거를 LLM용 한국어 프롬프트로 변환합니다."""

from __future__ import annotations

from typing import Any, Sequence


def build_rag_prompt(user_query: str, documents: Sequence[dict[str, Any]]) -> str:
    """문서 밖 추측과 충돌 자료의 임의 선택을 금지합니다."""
    sections = []
    for index, document in enumerate(documents, start=1):
        sections.append(
            f"[문서 {index}]\n"
            f"기존 질문: {document.get('retrieved_question', '')}\n"
            f"기존 답변: {document.get('retrieved_answer', '')}\n"
            f"인텐트: {document.get('intent', '')}"
        )
    references = "\n\n".join(sections)
    return f"""당신은 슈퍼 고객상담 도우미입니다.

아래 참고자료만 근거로 답변하세요.

규칙:
1. 참고자료에 있는 내용만 사용하세요.
2. 참고자료에 없는 정보는 추측하지 마세요.
3. 서로 다른 참고자료의 가격, 수량, 날짜, 조건이 충돌하면 하나를 임의로 선택하지 말고 자료가 달라 정확한 안내가 어렵다고 말하세요.
4. 상품명, 가격, 수량, 매장 상황 또는 대화 맥락이 불분명하면 추가 정보를 요청하세요.
5. 답을 확인할 수 없으면 \"제공된 상담 자료만으로는 확인하기 어렵습니다.\"라고 답하세요.
6. 간결하고 친절한 한국어로 답하세요.
7. 최종 답변 뒤에 참고한 문서 번호를 표시하세요.

사용자 질문: {user_query}

참고자료:
{references}
"""

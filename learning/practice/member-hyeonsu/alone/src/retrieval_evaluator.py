"""Intent Hit@k와 Intent MRR 계산 및 실패 후보 분류."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


def normalize_intent(value: object) -> str:
    """결측 가능 인텐트를 비교 가능한 문자열로 정리합니다."""
    return "" if value is None else str(value).strip()


@dataclass(frozen=True)
class QueryMetrics:
    """질문 하나의 인텐트 검색 평가 결과."""

    first_matching_rank: int | None
    hit_at_1: int
    hit_at_3: int
    hit_at_5: int
    reciprocal_rank: float


def evaluate_query(expected_intent: str, results: Sequence[dict[str, Any]]) -> QueryMetrics:
    """같은 인텐트가 처음 검색된 순위로 질문 하나의 지표를 계산합니다."""
    expected = normalize_intent(expected_intent)
    first_rank: int | None = None
    if expected:
        for result in results:
            retrieved = normalize_intent(result.get("metadata", {}).get("intent"))
            if retrieved == expected:
                first_rank = int(result["rank"])
                break
    return QueryMetrics(
        first_matching_rank=first_rank,
        hit_at_1=int(first_rank is not None and first_rank <= 1),
        hit_at_3=int(first_rank is not None and first_rank <= 3),
        hit_at_5=int(first_rank is not None and first_rank <= 5),
        reciprocal_rank=0.0 if first_rank is None else 1.0 / first_rank,
    )


def classify_failure_candidate(
    query: str,
    expected_intent: str,
    top1_intent: str,
    first_matching_rank: int | None,
) -> tuple[str, str]:
    """실패 원인을 확정하지 않고 사람이 볼 임시 후보만 규칙으로 지정합니다."""
    text = query.strip()
    expected = normalize_intent(expected_intent)
    top1 = normalize_intent(top1_intent)
    if len(text) <= 10:
        candidate = "짧은 질문"
    elif any(token in text for token in ("이거", "그거", "저거", "이 제품", "그 제품", "그걸")):
        candidate = "대화 맥락 부족 가능성"
    elif expected and top1 and expected.split("_")[0] == top1.split("_")[0]:
        candidate = "유사 인텐트 혼동"
    elif first_matching_rank is None:
        candidate = "Training 표본 부족 가능성"
    elif first_matching_rank > 5:
        candidate = "Training 표본 부족 가능성"
    elif not any(char.isdigit() for char in text) and any(
        token in text for token in ("상품", "제품", "가격", "재고", "배송")
    ):
        candidate = "상품명 또는 개체 정보 부족"
    else:
        candidate = "기타 검토 필요"
    return candidate, "규칙 기반 임시 후보이며 사람이 검색 결과와 원문을 확인해 최종 판단해야 함"

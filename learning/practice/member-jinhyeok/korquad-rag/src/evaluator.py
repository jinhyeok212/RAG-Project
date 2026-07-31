# Hit@k
# Reciprocal Rank
# 정답 문자열이 검색 Context에 존재하는가

def calculate_hit_at_k(
    retrieved_chunks: list[dict],
    ground_truth_doc_id: str
) -> int:
    """
    정답 문서가 Top-k 안에 있으면 1,
    없으면 0을 반환한다.
    """

    retrieved_doc_ids = {
        result["doc_id"]
        for result in retrieved_chunks
    }

    return int(
        ground_truth_doc_id in retrieved_doc_ids
    )


def calculate_reciprocal_rank(
    retrieved_chunks: list[dict],
    ground_truth_doc_id: str
) -> float:
    """
    정답 문서의 검색 순위를 이용해
    Reciprocal Rank를 계산한다.

    1위: 1 / 1 = 1.0
    2위: 1 / 2 = 0.5
    3위: 1 / 3 = 0.333...
    검색 실패: 0.0
    """

    for result in retrieved_chunks:
        if result["doc_id"] == ground_truth_doc_id:
            return 1.0 / result["rank"]

    return 0.0


def is_answer_in_context(
    ground_truth_answer: str,
    retrieved_chunks: list[dict]
) -> bool:
    """
    정답 문자열이 실제 검색된 Chunk 안에 존재하는지 확인한다.
    """

    normalized_answer = (
        ground_truth_answer
        .replace(" ", "")
        .strip()
    )

    combined_context = "".join(
        result["text"].replace(" ", "")
        for result in retrieved_chunks
    )

    return (
        normalized_answer in combined_context
        if normalized_answer
        else False
    )


def classify_failure(
    hit_at_k: int,
    answer_in_context: bool,
    generated_answer_correct: bool | None = None
) -> str:
    """
    단순 규칙으로 실패 유형을 분류한다.

    generated_answer_correct는 초기에는
    사람이 직접 확인한 후 전달할 수 있다.
    """

    if hit_at_k == 0:
        return "RETRIEVAL_FAILURE"

    if not answer_in_context:
        return "EVIDENCE_MISSING"

    if generated_answer_correct is False:
        return "GENERATION_FAILURE"

    if generated_answer_correct is True:
        return "SUCCESS"

    return "RETRIEVAL_SUCCESS_NOT_REVIEWED"
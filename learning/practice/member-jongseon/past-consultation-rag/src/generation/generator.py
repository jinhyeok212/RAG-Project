from __future__ import annotations

import re


GENERIC_ANSWER_PATTERNS = [
    "잠시만",
    "기다려",
    "준비해서",
    "확인해",
    "확인 후",
    "도와드리",
]

PRICE_QUESTION_PATTERNS = ["가격", "얼마", "금액", "비용"]
PRICE_ANSWER_RE = re.compile(r"(\d[\d,]*\s*(원|만\s*원)|[일이삼사오육칠팔구십천만]+\s*원)")


def _answer_quality_score(query: str, doc: dict, rank_index: int) -> float:
    answer = doc.get("answer_original") or ""
    intent = doc.get("intent") or ""
    score = 1.0 / (rank_index + 1)

    if any(pattern in answer for pattern in GENERIC_ANSWER_PATTERNS):
        score -= 2.0

    is_price_question = any(pattern in query for pattern in PRICE_QUESTION_PATTERNS) or "가격" in intent
    if is_price_question and PRICE_ANSWER_RE.search(answer):
        score += 3.0
    if "박스" in query and ("박스" in answer or "박스" in (doc.get("question_original") or "")):
        score += 0.5

    if answer.strip():
        score += min(len(answer), 80) / 1000.0

    return score


def generate_answer(query: str, retrieved_docs: list[dict], docs: dict[str, dict]) -> dict:
    if not retrieved_docs:
        return {
            "answer": "검색된 과거 상담 사례만으로는 확인할 수 없습니다.",
            "grounded": False,
            "insufficient_context": True,
            "sources": [],
            "retrieval_debug": {},
        }

    scored = []
    for idx, item in enumerate(retrieved_docs):
        doc = docs.get(item["doc_id"], {})
        scored.append((_answer_quality_score(query, doc, idx), idx, item, doc))
    scored.sort(key=lambda row: row[0], reverse=True)

    selected_score, selected_idx, selected_item, selected_doc = scored[0]
    answer = selected_doc.get("answer_original") or "검색된 상담 사례에 답변이 없습니다."

    sources = []
    for item in retrieved_docs:
        source = docs.get(item["doc_id"], {})
        sources.append(
            {
                "doc_id": item["doc_id"],
                "category": source.get("category"),
                "intent": source.get("intent"),
                "question": source.get("question_original"),
                "answer": source.get("answer_original"),
                "final_score": item.get("reranker_score") or item.get("rrf_score") or item.get("dense_score") or item.get("bm25_score"),
                "selected_for_answer": item["doc_id"] == selected_item["doc_id"],
            }
        )
    return {
        "answer": answer,
        "grounded": True,
        "insufficient_context": False,
        "sources": sources,
        "retrieval_debug": {
            "mode": "local_template_quality_aware_retrieved_answer",
            "selected_doc_id": selected_item["doc_id"],
            "selected_source_rank": selected_idx + 1,
            "selected_answer_quality_score": selected_score,
        },
    }

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DOCUMENTS = Path(r"C:\말똥가리\data\processed\ontong_youth_mvp_400_documents\documents.jsonl")
DEFAULT_OUTPUT_DIR = Path(r"C:\말똥가리\data\processed\ontong_youth_eval_questions_v1")

TARGET_COUNTS = {
    "일자리": 35,
    "복지문화": 20,
    "주거": 15,
    "교육": 15,
    "참여권리": 15,
}

QUESTION_TYPES = [
    "situation_search",
    "support_content",
    "eligibility",
    "application_method",
    "application_period",
    "required_documents",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def text(value: Any) -> str:
    return str(value or "").strip()


def compact(value: Any, limit: int = 500) -> str:
    cleaned = re.sub(r"\s+", " ", text(value))
    return cleaned[:limit]


def raw(doc: dict[str, Any], field: str) -> str:
    return text(doc.get("raw_record", {}).get(field, ""))


def category(doc: dict[str, Any]) -> str:
    return text(doc.get("metadata", {}).get("primary_category"))


def middle(doc: dict[str, Any]) -> str:
    return text(doc.get("metadata", {}).get("primary_mclsf"))


def keywords(doc: dict[str, Any]) -> list[str]:
    values = doc.get("metadata", {}).get("normalized_keywords") or []
    if isinstance(values, list):
        return [text(value) for value in values if text(value)]
    return []


def representative_need(doc: dict[str, Any]) -> str:
    cat = category(doc)
    mid = middle(doc)
    kws = set(keywords(doc))
    if cat == "주거":
        if "전월세" in mid or "주거지원" in kws:
            return "월세나 주거비 부담이 있는 청년"
        if "기숙사" in mid:
            return "거주 공간이나 기숙사가 필요한 청년"
        return "안정적인 주거 지원이 필요한 청년"
    if cat == "교육":
        if "교육비" in mid or "교육지원" in kws:
            return "교육비나 학자금 부담이 있는 청년"
        return "역량을 키우고 싶은 청년"
    if cat == "일자리":
        if "창업" in mid or "벤처" in kws:
            return "창업을 준비하거나 사업을 키우려는 청년"
        if "재직자" in mid:
            return "중소기업에 재직 중인 청년"
        return "취업이나 일자리 지원이 필요한 청년"
    if cat == "복지문화":
        if "건강" in mid:
            return "건강이나 상담 지원이 필요한 청년"
        if "문화" in mid:
            return "문화생활이나 생활지원을 받고 싶은 청년"
        return "금융이나 복지 지원이 필요한 청년"
    if cat == "참여권리":
        if "권익" in mid:
            return "권익 보호나 상담이 필요한 청년"
        return "정책 참여나 교류 기회를 찾는 청년"
    return "청년 정책 지원이 필요한 사람"


def has_field(doc: dict[str, Any], field: str) -> bool:
    return bool(raw(doc, field))


def candidate_question_types(doc: dict[str, Any]) -> list[str]:
    possible = ["situation_search", "support_content"]
    if has_field(doc, "addAplyQlfcCndCn") or has_field(doc, "ptcpPrpTrgtCn"):
        possible.append("eligibility")
    if has_field(doc, "plcyAplyMthdCn"):
        possible.append("application_method")
    if has_field(doc, "aplyYmd"):
        possible.append("application_period")
    if has_field(doc, "sbmsnDcmntCn"):
        possible.append("required_documents")
    return possible


def make_question(doc: dict[str, Any], qtype: str) -> tuple[str, str, str]:
    title = text(doc.get("title"))
    need = representative_need(doc)
    if qtype == "situation_search":
        return (
            f"{need}이 받을 만한 정책이 있나요?",
            "retrieval_text",
            compact(doc.get("retrieval_text"), 700),
        )
    if qtype == "support_content":
        return (
            f"{title}은 어떤 지원을 제공하나요?",
            "plcySprtCn",
            compact(raw(doc, "plcySprtCn"), 700),
        )
    if qtype == "eligibility":
        answer = " ".join(part for part in [raw(doc, "addAplyQlfcCndCn"), raw(doc, "ptcpPrpTrgtCn")] if part)
        return (
            f"{title}은 누가 신청할 수 있나요?",
            "addAplyQlfcCndCn+ptcpPrpTrgtCn",
            compact(answer, 700),
        )
    if qtype == "application_method":
        return (
            f"{title}은 어떻게 신청하나요?",
            "plcyAplyMthdCn",
            compact(raw(doc, "plcyAplyMthdCn"), 700),
        )
    if qtype == "application_period":
        return (
            f"{title}의 신청 기간은 언제인가요?",
            "aplyYmd",
            compact(raw(doc, "aplyYmd"), 300),
        )
    if qtype == "required_documents":
        return (
            f"{title}을 신청할 때 필요한 서류는 무엇인가요?",
            "sbmsnDcmntCn",
            compact(raw(doc, "sbmsnDcmntCn"), 700),
        )
    raise ValueError(f"unknown question type: {qtype}")


def document_sort_key(doc: dict[str, Any]) -> tuple[int, str]:
    score = int(doc.get("metadata", {}).get("document_quality_score") or 0)
    return (-score, text(doc.get("document_id")))


def select_docs_by_category(docs: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    by_middle = defaultdict(list)
    for doc in docs:
        by_middle[middle(doc) or "UNKNOWN"].append(doc)
    for values in by_middle.values():
        values.sort(key=document_sort_key)
    middles = sorted(by_middle, key=lambda key: (-len(by_middle[key]), key))
    selected: list[dict[str, Any]] = []
    selected_ids = set()
    while len(selected) < target:
        progressed = False
        for mid in middles:
            while by_middle[mid] and by_middle[mid][0].get("document_id") in selected_ids:
                by_middle[mid].pop(0)
            if not by_middle[mid]:
                continue
            doc = by_middle[mid].pop(0)
            selected.append(doc)
            selected_ids.add(doc.get("document_id"))
            progressed = True
            if len(selected) >= target:
                break
        if not progressed:
            break
    return selected


def run(documents_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    docs = read_jsonl(documents_path)
    by_category = defaultdict(list)
    for doc in docs:
        by_category[category(doc)].append(doc)

    selected_docs: list[dict[str, Any]] = []
    for cat, target in TARGET_COUNTS.items():
        selected_docs.extend(select_docs_by_category(by_category[cat], target))

    question_type_cycle = {
        "일자리": ["situation_search", "support_content", "eligibility", "application_method", "required_documents", "application_period"],
        "복지문화": ["situation_search", "support_content", "eligibility", "application_method", "required_documents", "application_period"],
        "주거": ["situation_search", "support_content", "eligibility", "application_period", "application_method", "required_documents"],
        "교육": ["situation_search", "support_content", "eligibility", "application_method", "application_period", "required_documents"],
        "참여권리": ["situation_search", "support_content", "application_method", "eligibility", "application_period", "required_documents"],
    }
    type_offsets = Counter()
    questions: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for doc in selected_docs:
        cat = category(doc)
        possible = candidate_question_types(doc)
        preferred = question_type_cycle[cat]
        qtype = ""
        for _ in range(len(preferred)):
            idx = type_offsets[cat] % len(preferred)
            type_offsets[cat] += 1
            if preferred[idx] in possible:
                qtype = preferred[idx]
                break
        if not qtype:
            skipped.append({"document_id": doc.get("document_id"), "title": doc.get("title"), "reason": "no_question_type"})
            continue
        query, answer_field, reference_answer = make_question(doc, qtype)
        if not query or not reference_answer:
            skipped.append({"document_id": doc.get("document_id"), "title": doc.get("title"), "reason": "blank_query_or_answer"})
            continue
        query_id = f"oy_eval_q{len(questions) + 1:04d}"
        question = {
            "query_id": query_id,
            "query": query,
            "ground_truth_document_ids": [doc["document_id"]],
            "question_type": qtype,
            "answer_source_field": answer_field,
            "reference_answer": reference_answer,
            "source_document_id": doc["document_id"],
            "source_title": doc["title"],
            "source_url": doc.get("url", ""),
            "primary_category": cat,
            "primary_mclsf": middle(doc),
            "normalized_keywords": keywords(doc),
            "human_review_label": "",
            "human_review_note": "",
        }
        questions.append(question)
        qrels.append(
            {
                "query_id": query_id,
                "document_id": doc["document_id"],
                "relevance": 1,
                "source": "human_to_review_candidate",
            }
        )

    review_fields = [
        "query_id",
        "query",
        "source_title",
        "source_document_id",
        "primary_category",
        "primary_mclsf",
        "question_type",
        "answer_source_field",
        "reference_answer",
        "source_url",
        "human_review_label",
        "human_review_note",
    ]
    write_jsonl(output_dir / "eval_question_candidates.jsonl", questions)
    write_csv(output_dir / "eval_question_candidates_review.csv", questions, review_fields)
    write_jsonl(output_dir / "qrels_candidates.jsonl", qrels)
    write_csv(output_dir / "qrels_candidates.csv", qrels, ["query_id", "document_id", "relevance", "source"])
    write_csv(output_dir / "eval_question_generation_skipped.csv", skipped, ["document_id", "title", "reason"])

    category_counts = Counter(q["primary_category"] for q in questions)
    type_counts = Counter(q["question_type"] for q in questions)
    middle_counts = Counter(q["primary_mclsf"] for q in questions)
    manifest = {
        "dataset_name": "Ontong Youth MVP 400 Evaluation Question Candidates",
        "dataset_version": "ontong_youth_eval_questions_v1_candidates",
        "source_documents": str(documents_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "HUMAN_REVIEW_REQUIRED",
        "question_count": len(questions),
        "qrels_count": len(qrels),
        "skipped_count": len(skipped),
        "target_counts": TARGET_COUNTS,
        "category_counts": dict(category_counts),
        "question_type_counts": dict(type_counts),
        "middle_counts": dict(middle_counts),
        "review_label_definition": {
            "1": "평가 질문으로 사용",
            "0": "평가 질문에서 제외",
            "2": "수정/보류 필요",
        },
        "output_files": {
            "eval_question_candidates_jsonl": str(output_dir / "eval_question_candidates.jsonl"),
            "eval_question_candidates_review_csv": str(output_dir / "eval_question_candidates_review.csv"),
            "qrels_candidates_jsonl": str(output_dir / "qrels_candidates.jsonl"),
            "qrels_candidates_csv": str(output_dir / "qrels_candidates.csv"),
            "skipped_csv": str(output_dir / "eval_question_generation_skipped.csv"),
            "manifest": str(output_dir / "eval_question_generation_manifest.json"),
        },
    }
    write_json(output_dir / "eval_question_generation_manifest.json", manifest)

    schema_md = """# Ontong Youth Evaluation Question Candidate Schema

## 목적

MVP 400개 정책 문서에서 retrieval 평가용 질문 후보를 생성한다. 이 파일은 최종 평가셋이 아니며, 사람이 검수해야 한다.

## 주요 파일

- `eval_question_candidates_review.csv`: 사람이 검수할 CSV
- `eval_question_candidates.jsonl`: 후보 질문 JSONL
- `qrels_candidates.jsonl`: 후보 질문과 정답 문서 연결

## 중요 필드

- `query_id`: 질문 ID
- `query`: 사용자 질문 형태의 평가 질문
- `ground_truth_document_ids`: 정답 문서 ID 배열
- `question_type`: 질문 유형
- `reference_answer`: 정답 문서에서 가져온 참고 답변. LLM 평가 정답이 아니라 검수 참고용
- `human_review_label`: 사람이 입력할 검수 라벨

## 검수 라벨

- `1`: 평가 질문으로 사용
- `0`: 평가 질문에서 제외
- `2`: 수정/보류 필요

## 주의

아직 chunking, embedding, Chroma, retrieval 평가는 수행하지 않았다.
"""
    (output_dir / "EVAL_QUESTION_SCHEMA.md").write_text(schema_md, encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = run(args.documents, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

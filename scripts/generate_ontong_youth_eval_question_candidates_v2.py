from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DOCUMENTS = Path(r"C:\말똥가리\data\processed\ontong_youth_mvp_400_documents\documents.jsonl")
DEFAULT_OUTPUT_DIR = Path(r"C:\말똥가리\data\processed\ontong_youth_eval_questions_v2")

TARGET_COUNTS = {
    "일자리": 35,
    "복지문화": 20,
    "주거": 15,
    "교육": 15,
    "참여권리": 15,
}

BAD_REFERENCE_PATTERNS = [
    "-",
    "해당없음",
    "없음",
    "해당 없음",
    "붙임파일",
    "공고문",
    "참고사이트",
    "홈페이지",
    "자세한 내용",
    "참조",
    "문의",
]

TITLE_STOPWORDS = {
    "지원",
    "사업",
    "청년",
    "모집",
    "공고",
    "2025년",
    "2026년",
    "프로그램",
}


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


def clean(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compact(value: Any, limit: int = 700) -> str:
    return re.sub(r"\s+", " ", clean(value))[:limit].strip()


def raw(doc: dict[str, Any], field: str) -> str:
    return clean(doc.get("raw_record", {}).get(field, ""))


def meta(doc: dict[str, Any], field: str) -> Any:
    return doc.get("metadata", {}).get(field, "")


def category(doc: dict[str, Any]) -> str:
    return str(meta(doc, "primary_category") or "").strip()


def middle(doc: dict[str, Any]) -> str:
    return str(meta(doc, "primary_mclsf") or "").strip()


def keywords(doc: dict[str, Any]) -> list[str]:
    values = meta(doc, "normalized_keywords") or []
    if isinstance(values, list):
        return [str(value).strip() for value in values if str(value).strip()]
    return []


def split_sentences(value: str) -> list[str]:
    text = clean(value)
    parts = re.split(r"(?<=[.!?。])\s+|\n+|ㆍ|·|•|○|□|※|①|②|③|④|⑤|⑥|⑦|⑧|⑨|- ", text)
    results = []
    for part in parts:
        part = re.sub(r"\s+", " ", part).strip(" -:;")
        if len(part) >= 8:
            results.append(part)
    return results


def good_reference(value: str, min_len: int = 12) -> bool:
    text = compact(value, 200)
    if len(text) < min_len:
        return False
    if text in BAD_REFERENCE_PATTERNS:
        return False
    bad_hits = sum(1 for pattern in BAD_REFERENCE_PATTERNS if pattern in text)
    if bad_hits and len(text) < 80:
        return False
    return True


def title_tokens(title: str) -> list[str]:
    tokens = []
    for token in re.split(r"[\s\[\]\(\)「」『』,·ㆍ_/]+", title):
        token = token.strip()
        if len(token) >= 3 and token not in TITLE_STOPWORDS and not re.fullmatch(r"\d{4}년?", token):
            tokens.append(token)
    return tokens


def title_overlap(query: str, title: str) -> int:
    return sum(1 for token in title_tokens(title) if token and token in query)


def extract_region(doc: dict[str, Any]) -> str:
    candidates = [
        doc.get("title", ""),
        meta(doc, "supervising_agency"),
        meta(doc, "operating_agency"),
        raw(doc, "addAplyQlfcCndCn"),
        raw(doc, "plcyExplnCn"),
    ]
    patterns = [
        r"[가-힣]+특별시",
        r"[가-힣]+광역시",
        r"[가-힣]+특별자치도",
        r"[가-힣]+특별자치시",
        r"[가-힣]+도",
        r"[가-힣]+시",
        r"[가-힣]+군",
        r"[가-힣]+구",
    ]
    for value in candidates:
        text = clean(value)
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                region = match.group(0)
                if region not in {"청년", "사업", "지원"} and len(region) <= 12:
                    return region
    return ""


def extract_target_phrase(doc: dict[str, Any]) -> str:
    fields = [raw(doc, "addAplyQlfcCndCn"), raw(doc, "plcySprtCn"), raw(doc, "plcyExplnCn")]
    for field in fields:
        for sent in split_sentences(field):
            if any(key in sent for key in ["청년", "재직", "취업", "창업", "대학생", "구직", "근로", "무주택", "농업", "예술", "중소기업"]):
                return compact(sent, 80)
    cat = category(doc)
    mid = middle(doc)
    if cat == "일자리":
        return "취업이나 창업을 준비하는 청년"
    if cat == "주거":
        return "주거비 부담이 있는 청년"
    if cat == "교육":
        return "교육이나 역량강화 기회가 필요한 청년"
    if cat == "복지문화":
        return "복지나 문화 지원이 필요한 청년"
    if cat == "참여권리":
        return "정책 참여나 권익 지원이 필요한 청년"
    return f"{mid} 지원이 필요한 청년"


def extract_benefit_phrase(doc: dict[str, Any]) -> str:
    support = raw(doc, "plcySprtCn")
    amount_pattern = r"[^.\n]{0,40}(?:월\s*)?(?:최대\s*)?\d[\d,]*(?:만)?원[^.\n]{0,50}"
    match = re.search(amount_pattern, support)
    if match:
        return compact(match.group(0), 90)
    for sent in split_sentences(support):
        if any(key in sent for key in ["지원", "지급", "제공", "대출", "보증", "교육", "상담", "포인트", "임차", "월세", "장학"]):
            return compact(sent, 90)
    return compact(support, 90)


def make_query_context(doc: dict[str, Any]) -> str:
    region = extract_region(doc)
    target = extract_target_phrase(doc)
    benefit = extract_benefit_phrase(doc)
    pieces = []
    if region and region not in target:
        pieces.append(region)
    if target:
        pieces.append(target)
    if benefit:
        pieces.append(benefit)
    context = " ".join(pieces)
    return compact(context, 140)


def make_question(doc: dict[str, Any], qtype: str) -> tuple[str, str, str]:
    context = make_query_context(doc)
    if qtype == "specific_policy_search":
        return (
            f"{context}와 관련된 정책을 찾으려면 어떤 문서를 봐야 하나요?",
            "retrieval_text",
            compact(doc.get("retrieval_text"), 800),
        )
    if qtype == "support_content":
        return (
            f"{context}의 지원 내용은 무엇인가요?",
            "plcySprtCn",
            compact(raw(doc, "plcySprtCn"), 800),
        )
    if qtype == "eligibility":
        answer = " ".join(part for part in [raw(doc, "addAplyQlfcCndCn"), raw(doc, "ptcpPrpTrgtCn")] if part)
        return (
            f"{context}에 해당하는 지원은 누가 신청할 수 있나요?",
            "addAplyQlfcCndCn+ptcpPrpTrgtCn",
            compact(answer, 800),
        )
    if qtype == "application_method":
        return (
            f"{context} 지원은 어떤 방식으로 신청하나요?",
            "plcyAplyMthdCn",
            compact(raw(doc, "plcyAplyMthdCn"), 800),
        )
    if qtype == "application_period":
        return (
            f"{context} 지원의 신청 기간은 언제인가요?",
            "aplyYmd",
            compact(raw(doc, "aplyYmd"), 300),
        )
    if qtype == "required_documents":
        return (
            f"{context} 지원을 신청할 때 준비해야 할 서류는 무엇인가요?",
            "sbmsnDcmntCn",
            compact(raw(doc, "sbmsnDcmntCn"), 800),
        )
    raise ValueError(qtype)


def possible_types(doc: dict[str, Any]) -> list[str]:
    possible = ["specific_policy_search", "support_content"]
    if good_reference(raw(doc, "addAplyQlfcCndCn") + " " + raw(doc, "ptcpPrpTrgtCn"), 30):
        possible.append("eligibility")
    if good_reference(raw(doc, "plcyAplyMthdCn"), 15):
        possible.append("application_method")
    if good_reference(raw(doc, "aplyYmd"), 8):
        possible.append("application_period")
    if good_reference(raw(doc, "sbmsnDcmntCn"), 20):
        possible.append("required_documents")
    return possible


def is_acceptable(doc: dict[str, Any], query: str, reference: str, qtype: str) -> tuple[bool, str]:
    if not good_reference(reference, 12):
        return False, "weak_reference_answer"
    if title_overlap(query, str(doc.get("title", ""))) >= 2:
        return False, "query_contains_too_much_policy_title"
    if len(query) < 25:
        return False, "query_too_short"
    if qtype == "specific_policy_search":
        context = make_query_context(doc)
        if len(context) < 35:
            return False, "specific_context_too_weak"
    if "받을 만한 정책이 있나요" in query:
        return False, "too_broad_phrase"
    return True, ""


def document_sort_key(doc: dict[str, Any]) -> tuple[int, str]:
    return (-int(meta(doc, "document_quality_score") or 0), str(doc.get("document_id")))


def select_docs_by_category(docs: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    by_middle = defaultdict(list)
    for doc in docs:
        by_middle[middle(doc) or "UNKNOWN"].append(doc)
    for values in by_middle.values():
        values.sort(key=document_sort_key)
    middles = sorted(by_middle, key=lambda key: (-len(by_middle[key]), key))
    selected = []
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


def generate_for_docs(selected_docs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    type_cycles = {
        "일자리": ["specific_policy_search", "support_content", "eligibility", "application_method", "required_documents", "application_period"],
        "복지문화": ["specific_policy_search", "support_content", "eligibility", "application_method", "required_documents", "application_period"],
        "주거": ["specific_policy_search", "support_content", "eligibility", "application_period", "application_method", "required_documents"],
        "교육": ["specific_policy_search", "support_content", "eligibility", "application_method", "application_period", "required_documents"],
        "참여권리": ["specific_policy_search", "support_content", "application_method", "eligibility", "application_period", "required_documents"],
    }
    offsets = Counter()
    questions = []
    qrels = []
    skipped = []
    used_docs = set()
    for doc in selected_docs:
        cat = category(doc)
        possible = possible_types(doc)
        chosen = ""
        query = answer_field = reference = ""
        for _ in range(len(type_cycles[cat])):
            idx = offsets[cat] % len(type_cycles[cat])
            offsets[cat] += 1
            qtype = type_cycles[cat][idx]
            if qtype not in possible:
                continue
            q, field, ref = make_question(doc, qtype)
            ok, reason = is_acceptable(doc, q, ref, qtype)
            if ok:
                chosen, query, answer_field, reference = qtype, q, field, ref
                break
        if not chosen:
            for qtype in possible:
                q, field, ref = make_question(doc, qtype)
                ok, reason = is_acceptable(doc, q, ref, qtype)
                if ok:
                    chosen, query, answer_field, reference = qtype, q, field, ref
                    break
        if not chosen:
            skipped.append({"document_id": doc.get("document_id"), "title": doc.get("title"), "reason": "no_acceptable_question"})
            continue
        query_id = f"oy_eval_v2_q{len(questions) + 1:04d}"
        item = {
            "query_id": query_id,
            "query": query,
            "ground_truth_document_ids": [doc["document_id"]],
            "question_type": chosen,
            "answer_source_field": answer_field,
            "reference_answer": reference,
            "source_document_id": doc["document_id"],
            "source_title": doc["title"],
            "source_url": doc.get("url", ""),
            "primary_category": cat,
            "primary_mclsf": middle(doc),
            "normalized_keywords": keywords(doc),
            "title_overlap_token_count": title_overlap(query, str(doc.get("title", ""))),
            "generation_rule_version": "v2_no_policy_title_specific_context",
            "human_review_label": "",
            "human_review_note": "",
        }
        questions.append(item)
        qrels.append({"query_id": query_id, "document_id": doc["document_id"], "relevance": 1, "source": "candidate_v2"})
        used_docs.add(doc["document_id"])
    return questions, qrels, skipped


def run(documents_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    docs = read_jsonl(documents_path)
    by_cat = defaultdict(list)
    for doc in docs:
        by_cat[category(doc)].append(doc)
    selected_docs = []
    for cat, target in TARGET_COUNTS.items():
        # Take extra documents because stricter rules can skip weak records.
        selected_docs.extend(select_docs_by_category(by_cat[cat], min(len(by_cat[cat]), target * 3)))
    questions, qrels, skipped = generate_for_docs(selected_docs)

    # Trim to exact category targets after strict generation.
    final_questions = []
    final_qrels = []
    cat_counts = Counter()
    for question in questions:
        cat = question["primary_category"]
        if cat_counts[cat] >= TARGET_COUNTS[cat]:
            continue
        cat_counts[cat] += 1
        new_id = f"oy_eval_v2_q{len(final_questions) + 1:04d}"
        old_id = question["query_id"]
        question = dict(question)
        question["query_id"] = new_id
        final_questions.append(question)
        final_qrels.append({"query_id": new_id, "document_id": question["source_document_id"], "relevance": 1, "source": "candidate_v2"})

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
        "title_overlap_token_count",
        "human_review_label",
        "human_review_note",
    ]
    write_jsonl(output_dir / "eval_question_candidates_v2.jsonl", final_questions)
    write_csv(output_dir / "eval_question_candidates_review_v2.csv", final_questions, review_fields)
    write_jsonl(output_dir / "qrels_candidates_v2.jsonl", final_qrels)
    write_csv(output_dir / "qrels_candidates_v2.csv", final_qrels, ["query_id", "document_id", "relevance", "source"])
    write_csv(output_dir / "eval_question_generation_skipped_v2.csv", skipped, ["document_id", "title", "reason"])

    manifest = {
        "dataset_name": "Ontong Youth MVP 400 Evaluation Question Candidates V2",
        "dataset_version": "ontong_youth_eval_questions_v2_candidates",
        "source_documents": str(documents_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "HUMAN_REVIEW_REQUIRED",
        "design_changes_from_v1": [
            "정책명을 질문에 그대로 넣는 템플릿 제거",
            "너무 포괄적인 '받을 만한 정책' 질문 제거",
            "지역/대상/조건/혜택 기반의 구체 질문 생성",
            "reference_answer가 '-' 또는 공고문 확인 수준이면 해당 유형 질문 제외",
            "질문과 정책명 title token overlap을 검사",
        ],
        "question_count": len(final_questions),
        "qrels_count": len(final_qrels),
        "skipped_count": len(skipped),
        "target_counts": TARGET_COUNTS,
        "category_counts": dict(Counter(q["primary_category"] for q in final_questions)),
        "question_type_counts": dict(Counter(q["question_type"] for q in final_questions)),
        "middle_counts": dict(Counter(q["primary_mclsf"] for q in final_questions)),
        "title_overlap_distribution": dict(Counter(str(q["title_overlap_token_count"]) for q in final_questions)),
        "review_label_definition": {
            "1": "평가 질문으로 사용",
            "0": "평가 질문에서 제외",
            "2": "수정/보류 필요",
        },
        "output_files": {
            "eval_question_candidates_jsonl": str(output_dir / "eval_question_candidates_v2.jsonl"),
            "eval_question_candidates_review_csv": str(output_dir / "eval_question_candidates_review_v2.csv"),
            "qrels_candidates_jsonl": str(output_dir / "qrels_candidates_v2.jsonl"),
            "qrels_candidates_csv": str(output_dir / "qrels_candidates_v2.csv"),
            "skipped_csv": str(output_dir / "eval_question_generation_skipped_v2.csv"),
            "manifest": str(output_dir / "eval_question_generation_manifest_v2.json"),
        },
    }
    write_json(output_dir / "eval_question_generation_manifest_v2.json", manifest)
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

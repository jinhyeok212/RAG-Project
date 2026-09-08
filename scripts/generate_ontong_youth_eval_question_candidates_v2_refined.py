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
DEFAULT_OUTPUT_DIR = Path(r"C:\말똥가리\data\processed\ontong_youth_eval_questions_v2_refined")

TARGET_COUNTS = {"일자리": 35, "복지문화": 20, "주거": 15, "교육": 15, "참여권리": 15}

BAD_SHORT_REFERENCES = {"-", "없음", "해당없음", "해당 없음", "첨부파일 참조", "공고문 참조"}
WEAK_REFERENCE_HINTS = ["붙임파일", "공고문", "참고사이트", "자세한 내용", "문의", "홈페이지 확인"]
TITLE_STOPWORDS = {"지원", "사업", "청년", "모집", "공고", "프로그램", "참여자", "2025년", "2026년"}


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


def compact(value: Any, limit: int = 800) -> str:
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
    return [str(value).strip() for value in values if str(value).strip()] if isinstance(values, list) else []


def full_text(doc: dict[str, Any]) -> str:
    return " ".join(
        raw(doc, field)
        for field in ["plcyNm", "plcyExplnCn", "plcySprtCn", "addAplyQlfcCndCn", "ptcpPrpTrgtCn", "plcyAplyMthdCn"]
    )


def title_tokens(title: str) -> list[str]:
    tokens = []
    for token in re.split(r"[\s\[\]\(\)「」『』,·ㆍ_/]+", title):
        token = token.strip()
        if len(token) >= 3 and token not in TITLE_STOPWORDS and not re.fullmatch(r"\d{4}년?", token):
            tokens.append(token)
    return tokens


def title_overlap(query: str, title: str) -> int:
    return sum(1 for token in title_tokens(title) if token in query)


def good_reference(value: str, min_len: int = 12) -> bool:
    text = compact(value, 200)
    if len(text) < min_len:
        return False
    if text in BAD_SHORT_REFERENCES:
        return False
    if len(text) < 80 and any(hint in text for hint in WEAK_REFERENCE_HINTS):
        return False
    return True


def extract_region(doc: dict[str, Any]) -> str:
    text = " ".join([str(doc.get("title", "")), str(meta(doc, "supervising_agency")), str(meta(doc, "operating_agency")), raw(doc, "plcyExplnCn")])
    aliases = [
        "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
    ]
    for alias in aliases:
        if alias in text:
            return alias
    match = re.search(r"([가-힣]{2,8})(?:시|군|구)", text)
    return match.group(0) if match else ""


def beneficiary(doc: dict[str, Any]) -> str:
    txt = full_text(doc)
    mid = middle(doc)
    if "대학생" in txt or "대학" in txt or "학자금" in txt:
        return "대학생 또는 졸업생"
    if "중소기업" in txt and ("재직" in txt or mid == "재직자"):
        return "중소기업 재직 청년"
    if "미취업" in txt or "구직" in txt:
        return "미취업 청년"
    if "창업" in txt or mid == "창업":
        return "창업을 준비하는 청년"
    if "농업" in txt or "농업인" in txt:
        return "청년 농업인"
    if "예술" in txt or "문화" in txt:
        return "문화예술 활동을 하는 청년"
    if "무주택" in txt or category(doc) == "주거":
        return "주거 지원이 필요한 청년"
    if "장애" in txt:
        return "장애가 있는 청년"
    if mid == "청년참여":
        return "정책 참여 기회를 찾는 청년"
    if mid == "권익보호":
        return "권익 보호가 필요한 청년"
    if category(doc) == "교육":
        return "교육이나 역량강화가 필요한 청년"
    if category(doc) == "복지문화":
        return "복지나 생활 지원이 필요한 청년"
    return "청년"


def benefit(doc: dict[str, Any]) -> str:
    txt = full_text(doc)
    if "월세" in txt or "주거비" in txt or "임차료" in txt or "전세" in txt:
        return "주거비 또는 임차료 지원"
    if "학자금" in txt or "장학" in txt:
        return "학자금 또는 장학금 지원"
    if "응시료" in txt or "자격증" in txt or "자격시험" in txt:
        return "자격시험 응시료 지원"
    if "복지포인트" in txt or "포인트" in txt:
        return "복지포인트 지원"
    if "상품권" in txt:
        return "상품권 지원"
    if "보증" in txt or "대출" in txt:
        return "대출 또는 보증 지원"
    if "인턴" in txt:
        return "인턴 실무경험 지원"
    if "교육" in txt or "훈련" in txt:
        return "교육 또는 훈련 지원"
    if "상담" in txt or "컨설팅" in txt:
        return "상담 또는 컨설팅 지원"
    if "창업" in txt:
        return "창업 지원"
    if "취업" in txt:
        return "취업 지원"
    if "문화" in txt:
        return "문화활동 지원"
    return f"{middle(doc)} 관련 지원"


def context_phrase(doc: dict[str, Any]) -> str:
    region = extract_region(doc)
    who = beneficiary(doc)
    what = benefit(doc)
    if region:
        return f"{region}에서 {who}을 위한 {what}"
    return f"{who}을 위한 {what}"


def make_question(doc: dict[str, Any], qtype: str) -> tuple[str, str, str]:
    ctx = context_phrase(doc)
    if qtype == "specific_policy_search":
        return f"{ctx}을 찾을 때 확인해야 할 정책 문서는 무엇인가요?", "retrieval_text", compact(doc.get("retrieval_text"), 800)
    if qtype == "support_content":
        return f"{ctx}의 구체적인 지원 내용은 무엇인가요?", "plcySprtCn", compact(raw(doc, "plcySprtCn"), 800)
    if qtype == "eligibility":
        answer = " ".join(part for part in [raw(doc, "addAplyQlfcCndCn"), raw(doc, "ptcpPrpTrgtCn")] if part)
        return f"{ctx}을 신청하려면 어떤 자격 조건이 필요한가요?", "addAplyQlfcCndCn+ptcpPrpTrgtCn", compact(answer, 800)
    if qtype == "application_method":
        return f"{ctx}은 어떤 방식으로 신청하나요?", "plcyAplyMthdCn", compact(raw(doc, "plcyAplyMthdCn"), 800)
    if qtype == "application_period":
        return f"{ctx}의 신청 기간은 언제인가요?", "aplyYmd", compact(raw(doc, "aplyYmd"), 300)
    if qtype == "required_documents":
        return f"{ctx}을 신청할 때 어떤 서류를 준비해야 하나요?", "sbmsnDcmntCn", compact(raw(doc, "sbmsnDcmntCn"), 800)
    raise ValueError(qtype)


def possible_types(doc: dict[str, Any]) -> list[str]:
    possible = ["specific_policy_search", "support_content"]
    if good_reference(raw(doc, "addAplyQlfcCndCn") + " " + raw(doc, "ptcpPrpTrgtCn"), 30):
        possible.append("eligibility")
    if good_reference(raw(doc, "plcyAplyMthdCn"), 15):
        possible.append("application_method")
    if good_reference(raw(doc, "aplyYmd"), 8):
        possible.append("application_period")
    if good_reference(raw(doc, "sbmsnDcmntCn"), 25):
        possible.append("required_documents")
    return possible


def acceptable(doc: dict[str, Any], query: str, reference: str, qtype: str) -> tuple[bool, str]:
    if title_overlap(query, str(doc.get("title", ""))) >= 2:
        return False, "title_overlap_too_high"
    if not good_reference(reference):
        return False, "weak_reference"
    if len(query) > 95:
        return False, "query_too_long"
    if "받을 만한 정책" in query:
        return False, "too_broad"
    return True, ""


def document_sort_key(doc: dict[str, Any]) -> tuple[int, str]:
    return (-int(meta(doc, "document_quality_score") or 0), str(doc.get("document_id")))


def select_docs_by_category(docs: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    by_middle = defaultdict(list)
    for doc in docs:
        by_middle[middle(doc) or "UNKNOWN"].append(doc)
    for values in by_middle.values():
        values.sort(key=document_sort_key)
    mids = sorted(by_middle, key=lambda key: (-len(by_middle[key]), key))
    selected, seen = [], set()
    while len(selected) < target and mids:
        progressed = False
        for mid in mids:
            while by_middle[mid] and by_middle[mid][0].get("document_id") in seen:
                by_middle[mid].pop(0)
            if by_middle[mid]:
                doc = by_middle[mid].pop(0)
                selected.append(doc)
                seen.add(doc.get("document_id"))
                progressed = True
                if len(selected) >= target:
                    break
        if not progressed:
            break
    return selected


def run(documents_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    docs = read_jsonl(documents_path)
    by_cat = defaultdict(list)
    for doc in docs:
        by_cat[category(doc)].append(doc)

    selected_docs = []
    for cat, target in TARGET_COUNTS.items():
        selected_docs.extend(select_docs_by_category(by_cat[cat], target * 3))

    cycles = {
        "일자리": ["specific_policy_search", "support_content", "eligibility", "application_method", "required_documents", "application_period"],
        "복지문화": ["specific_policy_search", "support_content", "eligibility", "application_method", "required_documents", "application_period"],
        "주거": ["specific_policy_search", "support_content", "eligibility", "application_period", "application_method", "required_documents"],
        "교육": ["specific_policy_search", "support_content", "eligibility", "application_method", "application_period", "required_documents"],
        "참여권리": ["specific_policy_search", "support_content", "application_method", "eligibility", "application_period", "required_documents"],
    }
    offsets = Counter()
    questions, skipped = [], []
    cat_counts = Counter()
    used_docs = set()
    for doc in selected_docs:
        cat = category(doc)
        if cat_counts[cat] >= TARGET_COUNTS[cat]:
            continue
        qtype = ""
        query = field = ref = ""
        possible = possible_types(doc)
        for _ in range(len(cycles[cat])):
            idx = offsets[cat] % len(cycles[cat])
            offsets[cat] += 1
            candidate_type = cycles[cat][idx]
            if candidate_type not in possible:
                continue
            q, f, r = make_question(doc, candidate_type)
            ok, reason = acceptable(doc, q, r, candidate_type)
            if ok:
                qtype, query, field, ref = candidate_type, q, f, r
                break
        if not qtype:
            for candidate_type in possible:
                q, f, r = make_question(doc, candidate_type)
                ok, reason = acceptable(doc, q, r, candidate_type)
                if ok:
                    qtype, query, field, ref = candidate_type, q, f, r
                    break
        if not qtype:
            skipped.append({"document_id": doc.get("document_id"), "title": doc.get("title"), "reason": "no_acceptable_question"})
            continue
        cat_counts[cat] += 1
        query_id = f"oy_eval_v2r_q{len(questions) + 1:04d}"
        questions.append(
            {
                "query_id": query_id,
                "query": query,
                "ground_truth_document_ids": [doc["document_id"]],
                "question_type": qtype,
                "answer_source_field": field,
                "reference_answer": ref,
                "source_document_id": doc["document_id"],
                "source_title": doc["title"],
                "source_url": doc.get("url", ""),
                "primary_category": cat,
                "primary_mclsf": middle(doc),
                "normalized_keywords": keywords(doc),
                "title_overlap_token_count": title_overlap(query, str(doc.get("title", ""))),
                "generation_rule_version": "v2_refined_short_context_no_title",
                "human_review_label": "",
                "human_review_note": "",
            }
        )
        used_docs.add(doc["document_id"])

    qrels = [{"query_id": q["query_id"], "document_id": q["source_document_id"], "relevance": 1, "source": "candidate_v2_refined"} for q in questions]
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
    write_jsonl(output_dir / "eval_question_candidates_v2_refined.jsonl", questions)
    write_csv(output_dir / "eval_question_candidates_review_v2_refined.csv", questions, review_fields)
    write_jsonl(output_dir / "qrels_candidates_v2_refined.jsonl", qrels)
    write_csv(output_dir / "qrels_candidates_v2_refined.csv", qrels, ["query_id", "document_id", "relevance", "source"])
    write_csv(output_dir / "eval_question_generation_skipped_v2_refined.csv", skipped, ["document_id", "title", "reason"])
    manifest = {
        "dataset_name": "Ontong Youth MVP 400 Evaluation Question Candidates V2 Refined",
        "dataset_version": "ontong_youth_eval_questions_v2_refined_candidates",
        "source_documents": str(documents_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "HUMAN_REVIEW_REQUIRED",
        "question_count": len(questions),
        "qrels_count": len(qrels),
        "skipped_count": len(skipped),
        "target_counts": TARGET_COUNTS,
        "category_counts": dict(Counter(q["primary_category"] for q in questions)),
        "question_type_counts": dict(Counter(q["question_type"] for q in questions)),
        "middle_counts": dict(Counter(q["primary_mclsf"] for q in questions)),
        "title_overlap_distribution": dict(Counter(str(q["title_overlap_token_count"]) for q in questions)),
        "design_changes": [
            "정책명 직접 사용 금지",
            "원문 조건/지원문구 긴 복붙 금지",
            "지역 + 대상 + 혜택유형 중심의 짧은 질문 생성",
            "reference_answer가 빈약한 제출서류/자격 질문 제외",
        ],
        "review_label_definition": {"1": "평가 질문으로 사용", "0": "평가 질문에서 제외", "2": "수정/보류 필요"},
        "output_files": {
            "review_csv": str(output_dir / "eval_question_candidates_review_v2_refined.csv"),
            "questions_jsonl": str(output_dir / "eval_question_candidates_v2_refined.jsonl"),
            "qrels_jsonl": str(output_dir / "qrels_candidates_v2_refined.jsonl"),
            "manifest": str(output_dir / "eval_question_generation_manifest_v2_refined.json"),
        },
    }
    write_json(output_dir / "eval_question_generation_manifest_v2_refined.json", manifest)
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

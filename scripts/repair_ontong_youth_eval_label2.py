from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ADJUDICATED = Path(
    r"C:\말똥가리\data\processed\ontong_youth_eval_questions_v2_refined\eval_question_candidates_review_v2_refined_adjudicated.csv"
)
DEFAULT_DOCUMENTS = Path(r"C:\말똥가리\data\processed\ontong_youth_mvp_400_documents\documents.jsonl")
DEFAULT_OUTPUT_DIR = Path(r"C:\말똥가리\data\processed\ontong_youth_eval_questions_v2_repaired")


# Repairs are intentionally explicit. These 33 rows were adjudicated as "2":
# usable after adding a missing slot, changing the subject, replacing evidence,
# or marking multi-document qrels.
REPAIR_MAP: dict[str, dict[str, Any]] = {
    "oy_eval_v2r_q0007": {
        "query": "대구에서 신산업 분야 창업경진대회에 참여하려는 청년 창업기업이 확인해야 할 정책 문서는 무엇인가요?",
        "strategy": "add_slot",
        "slot": "창업경진대회",
    },
    "oy_eval_v2r_q0009": {
        "query": "대구에서 미취업 청년이 국가기술자격시험 응시료 지원을 신청하려면 어떤 자격 조건이 필요한가요?",
        "strategy": "add_slot",
        "slot": "국가기술자격시험",
    },
    "oy_eval_v2r_q0012": {
        "query": "광주에서 마을 공익일자리 활동에 참여하려는 청년은 언제 신청해야 하나요?",
        "strategy": "add_slot",
        "slot": "마을 공익일자리",
    },
    "oy_eval_v2r_q0021": {
        "query": "세종에서 농업 분야 창업을 준비하는 청년은 영농정착 지원을 어떤 방식으로 신청하나요?",
        "strategy": "add_slot",
        "slot": "농업 분야",
    },
    "oy_eval_v2r_q0024": {
        "query": "울산에서 미취업 청년이 자격시험 응시료 지원을 찾을 때 확인해야 할 정책 문서는 무엇인가요?",
        "strategy": "add_slot",
        "slot": "울산",
    },
    "oy_eval_v2r_q0027": {
        "query": "경북에서 사업화자금 20~30백만원을 지원받는 청년 CEO 기업은 어떻게 신청하나요?",
        "strategy": "add_slot",
        "slot": "사업화자금 20~30백만원",
    },
    "oy_eval_v2r_q0030": {
        "query": "산림 분야 창업이나 취업을 준비하는 청년을 위한 교육 지원의 구체적인 내용은 무엇인가요?",
        "strategy": "add_slot",
        "slot": "산림 분야",
    },
    "oy_eval_v2r_q0032": {
        "query": "고졸 취업연계 장려금은 어떤 방식으로 신청하나요?",
        "strategy": "add_slot",
        "slot": "고졸 취업연계",
    },
    "oy_eval_v2r_q0033": {
        "query": "광주에서 G-유니콘 육성프로그램을 통해 사업화자금을 받으려는 창업기업의 신청 기간은 언제인가요?",
        "strategy": "add_slot",
        "slot": "G-유니콘/사업화자금",
    },
    "oy_eval_v2r_q0034": {
        "query": "강원에서 직무교육 후 채용연계를 받을 수 있는 미래인력 양성 정책 문서는 무엇인가요?",
        "strategy": "add_slot",
        "slot": "직무교육 후 채용연계",
    },
    "oy_eval_v2r_q0035": {
        "query": "광주에서 관광·MICE 분야 청년 창업기업을 위한 교육·컨설팅 지원 내용은 무엇인가요?",
        "strategy": "add_slot",
        "slot": "관광·MICE 분야",
    },
    "oy_eval_v2r_q0037": {
        "query": "서울에서 청년문화패스 문화이용권으로 받을 수 있는 구체적인 지원 내용은 무엇인가요?",
        "strategy": "needs_multi_qrels",
        "slot": "서울청년문화패스",
    },
    "oy_eval_v2r_q0038": {
        "query": "전국민 마음투자 바우처 상담 지원은 누가 신청할 수 있나요?",
        "strategy": "needs_multi_qrels",
        "slot": "전국민마음투자/바우처",
    },
    "oy_eval_v2r_q0039": {
        "query": "광주에서 청년 예술인이 창작활동비 지원을 신청하려면 어떤 방식으로 신청하나요?",
        "strategy": "add_slot",
        "slot": "창작활동비 지원",
    },
    "oy_eval_v2r_q0042": {
        "query": "청년이 건강관리와 모바일 헬스케어 상담을 받을 수 있는 정책 문서는 무엇인가요?",
        "strategy": "add_slot",
        "slot": "건강관리·모바일 헬스케어",
    },
    "oy_eval_v2r_q0043": {
        "query": "광주에서 청년예술인을 위한 문화활동 지원의 구체적인 내용은 무엇인가요?",
        "strategy": "replace_subject",
        "slot": "청년예술인",
    },
    "oy_eval_v2r_q0045": {
        "query": "세종 청년문화생태계 리빙랩의 대학가요제에 신청할 때 어떤 서류를 준비해야 하나요?",
        "strategy": "add_slot",
        "slot": "대학가요제",
    },
    "oy_eval_v2r_q0047": {
        "query": "전북에서 단편영화 제작스쿨을 운영하는 영화영상 지원 정책 문서는 무엇인가요?",
        "strategy": "add_slot",
        "slot": "단편영화 제작스쿨",
    },
    "oy_eval_v2r_q0054": {
        "query": "경기에서 청년 체력증진과 BMI 관리 서비스를 신청하려면 어떤 방식으로 신청하나요?",
        "strategy": "add_slot",
        "slot": "체력증진·BMI 관리 서비스",
    },
    "oy_eval_v2r_q0056": {
        "query": "경기에서 창업지원주택 같은 청년 창업자 주거 지원을 찾을 때 확인해야 할 정책 문서는 무엇인가요?",
        "strategy": "add_slot",
        "slot": "창업지원주택·주거",
    },
    "oy_eval_v2r_q0066": {
        "query": "세종에서 청년 주택임차보증금 대출이자 지원을 신청하려면 어떤 자격 조건이 필요한가요?",
        "strategy": "replace_evidence_needed",
        "slot": "임차보증금 대출이자",
    },
    "oy_eval_v2r_q0072": {
        "query": "소득연계 국가장학금은 대학생에게 어떤 지원을 제공하나요?",
        "strategy": "add_slot",
        "slot": "소득연계 국가장학금",
    },
    "oy_eval_v2r_q0073": {
        "query": "서울 청년인생설계학교에 참여하려면 어떤 신청 자격이 필요한가요?",
        "strategy": "replace_evidence_needed",
        "slot": "서울 청년인생설계학교 신청자격",
    },
    "oy_eval_v2r_q0074": {
        "query": "임업 분야 청년을 위한 특성화교육은 어떤 방식으로 신청하나요?",
        "strategy": "add_slot",
        "slot": "임업 분야",
    },
    "oy_eval_v2r_q0075": {
        "query": "대학 실용금융 강좌 지원의 신청 기간은 언제인가요?",
        "strategy": "add_slot",
        "slot": "대학 금융강좌",
    },
    "oy_eval_v2r_q0076": {
        "query": "예술·체육 분야 우수학생 국가장학금을 신청할 때 어떤 서류를 준비해야 하나요?",
        "strategy": "add_slot",
        "slot": "예술·체육 분야",
    },
    "oy_eval_v2r_q0077": {
        "query": "광주에서 평생교육이용권을 신청하려는 장애인은 어떤 정책 문서를 확인해야 하나요?",
        "strategy": "add_slot",
        "slot": "평생교육이용권",
    },
    "oy_eval_v2r_q0080": {
        "query": "광주 남구 청년 자격증 취득 지원은 어떤 방식으로 신청하나요?",
        "strategy": "needs_multi_qrels",
        "slot": "남구 청년 자격증",
    },
    "oy_eval_v2r_q0082": {
        "query": "대학생 금융교육봉사단에 지원할 때 어떤 서류를 준비해야 하나요?",
        "strategy": "add_slot",
        "slot": "금융교육 봉사단",
    },
    "oy_eval_v2r_q0085": {
        "query": "대학 실용금융 강좌 온라인(K-MOOC)은 어떤 방식으로 신청하나요?",
        "strategy": "add_slot",
        "slot": "온라인(K-MOOC)",
    },
    "oy_eval_v2r_q0089": {
        "query": "경기도 내 대학교가 노동인권 강좌 개설 지원을 신청할 수 있는 기간은 언제인가요?",
        "strategy": "replace_subject",
        "slot": "신청 주체를 대학으로 변경",
    },
    "oy_eval_v2r_q0099": {
        "query": "세종에서 온라인 정책제안 창구를 통해 청년정책 발굴 사업에 참여하려면 신청 기간은 언제인가요?",
        "strategy": "add_slot",
        "slot": "온라인 정책제안 창구",
    },
    "oy_eval_v2r_q0100": {
        "query": "에너지 분야 국제기구 인턴 파견에 지원할 때 어떤 서류를 준비해야 하나요?",
        "strategy": "add_slot",
        "slot": "에너지 분야",
    },
}


def read_csv_auto(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                return list(reader), reader.fieldnames or []
        except UnicodeDecodeError:
            continue
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), reader.fieldnames or []


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def value(row: dict[str, Any], key: str) -> str:
    item = row.get(key, "")
    return "" if item is None else str(item).strip()


def find_better_reference(doc: dict[str, Any], strategy: str, existing: str) -> str:
    if strategy != "replace_evidence_needed":
        return existing
    retrieval_text = str(doc.get("retrieval_text", ""))
    if retrieval_text:
        return retrieval_text[:900]
    return existing


def run(adjudicated_path: Path, documents_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, _ = read_csv_auto(adjudicated_path)
    docs = {doc["document_id"]: doc for doc in read_jsonl(documents_path)}
    label2_rows = [row for row in rows if value(row, "final_label") == "2"]

    repaired = []
    skipped = []
    for row in label2_rows:
        qid = value(row, "query_id")
        repair = REPAIR_MAP.get(qid)
        if not repair:
            skipped.append({"query_id": qid, "reason": "missing_manual_repair"})
            continue
        doc_id = value(row, "source_document_id")
        doc = docs.get(doc_id, {})
        repaired_query_id = qid.replace("oy_eval_v2r_", "oy_eval_repair_")
        repaired_reference = find_better_reference(doc, repair["strategy"], value(row, "reference_answer"))
        repaired.append(
            {
                "repair_query_id": repaired_query_id,
                "source_query_id": qid,
                "original_query": value(row, "query"),
                "repaired_query": repair["query"],
                "source_title": value(row, "source_title"),
                "source_document_id": doc_id,
                "ground_truth_document_ids": json.dumps([doc_id], ensure_ascii=False),
                "primary_category": value(row, "primary_category"),
                "primary_mclsf": value(row, "primary_mclsf"),
                "question_type": value(row, "question_type"),
                "answer_source_field": value(row, "answer_source_field"),
                "reference_answer": repaired_reference,
                "source_url": value(row, "source_url"),
                "original_final_note": value(row, "final_note"),
                "repair_strategy": repair["strategy"],
                "repair_slot": repair["slot"],
                "needs_multi_qrels_review": "Y" if repair["strategy"] == "needs_multi_qrels" else "N",
                "needs_reference_review": "Y" if repair["strategy"] == "replace_evidence_needed" else "N",
                "human_review_label": "",
                "human_review_note": "",
            }
        )

    review_fields = [
        "repair_query_id",
        "source_query_id",
        "original_query",
        "repaired_query",
        "source_title",
        "source_document_id",
        "ground_truth_document_ids",
        "primary_category",
        "primary_mclsf",
        "question_type",
        "answer_source_field",
        "reference_answer",
        "source_url",
        "original_final_note",
        "repair_strategy",
        "repair_slot",
        "needs_multi_qrels_review",
        "needs_reference_review",
        "human_review_label",
        "human_review_note",
    ]
    qrels = [
        {
            "query_id": row["repair_query_id"],
            "document_id": row["source_document_id"],
            "relevance": 1,
            "source": "label2_repair_candidate",
            "needs_multi_qrels_review": row["needs_multi_qrels_review"],
        }
        for row in repaired
    ]

    write_csv(output_dir / "eval_question_repair_candidates.csv", repaired, review_fields)
    write_jsonl(output_dir / "eval_question_repair_candidates.jsonl", repaired)
    write_jsonl(output_dir / "qrels_repair_candidates.jsonl", qrels)
    write_csv(
        output_dir / "qrels_repair_candidates.csv",
        qrels,
        ["query_id", "document_id", "relevance", "source", "needs_multi_qrels_review"],
    )
    write_csv(output_dir / "eval_question_repair_skipped.csv", skipped, ["query_id", "reason"])

    manifest = {
        "dataset_name": "Ontong Youth Eval Question Label2 Repair Candidates",
        "dataset_version": "ontong_youth_eval_question_repair_v1",
        "source_adjudicated_file": str(adjudicated_path),
        "source_documents": str(documents_path),
        "created_at": datetime.now().astimezone().isoformat(),
        "input_label2_count": len(label2_rows),
        "repaired_count": len(repaired),
        "skipped_count": len(skipped),
        "repair_strategy_counts": dict(Counter(row["repair_strategy"] for row in repaired)),
        "needs_multi_qrels_review_count": sum(1 for row in repaired if row["needs_multi_qrels_review"] == "Y"),
        "needs_reference_review_count": sum(1 for row in repaired if row["needs_reference_review"] == "Y"),
        "review_label_definition": {
            "1": "수정 질문을 최종 평가 질문으로 사용",
            "0": "수정해도 제외",
            "2": "아직 추가 수정/보류 필요",
        },
        "output_files": {
            "repair_review_csv": str(output_dir / "eval_question_repair_candidates.csv"),
            "repair_jsonl": str(output_dir / "eval_question_repair_candidates.jsonl"),
            "qrels_repair_jsonl": str(output_dir / "qrels_repair_candidates.jsonl"),
            "manifest": str(output_dir / "repair_manifest.json"),
        },
    }
    write_json(output_dir / "repair_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjudicated", type=Path, default=DEFAULT_ADJUDICATED)
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = run(args.adjudicated, args.documents, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import csv
import json
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = ROOT / "data/processed/ontong_youth_mvp_400_documents/documents.jsonl"
ADJUDICATED_PATH = ROOT / "data/processed/ontong_youth_eval_questions_v2_refined/eval_question_candidates_review_v2_refined_adjudicated_standardized.csv"
REPAIR_LABELED_PATH = ROOT / "data/processed/ontong_youth_eval_questions_v2_repaired/eval_question_repair_candidates_claude_labeled.csv"
OUT_DIR = ROOT / "data/processed/ontong_youth_eval_questions_v2_final_47"


KST = timezone(timedelta(hours=9))


RESOLVED_LABEL2 = {
    "oy_eval_repair_q0021": {
        "query": "세종시에서 자체 배정으로 선발되는 청년농업인(후계농) 영농정착 지원은 어떤 방식으로 신청하나요?",
        "ground_truth_document_ids": ["ontong_youth_20250909005400211682"],
        "resolution_action": "question_refined",
        "resolution_note": "전국 유사 사업과 구분되도록 '세종시 자체 배정' 조건을 추가해 세종 문서 단일 정답으로 유지.",
    },
    "oy_eval_repair_q0030": {
        "query": "산림 분야 취업이나 창업을 준비하는 청년을 위한 교육 프로그램 20종 공모·운영 지원의 구체적인 내용은 무엇인가요?",
        "ground_truth_document_ids": ["ontong_youth_20250529005400110881"],
        "resolution_action": "question_refined",
        "resolution_note": "산림청년포럼/시너지캠프와 구분되도록 '교육 프로그램 20종 공모·운영' 슬롯을 추가.",
    },
    "oy_eval_repair_q0032": {
        "query": "2025학년도 고교 취업연계 장려금은 어떤 방식으로 신청하나요?",
        "ground_truth_document_ids": ["ontong_youth_20250612005400110914"],
        "resolution_action": "question_refined",
        "resolution_note": "2026년 문서와 구분되도록 '2025학년도'를 명시.",
    },
    "oy_eval_repair_q0037": {
        "ground_truth_document_ids": [
            "ontong_youth_20250227005400210572",
            "ontong_youth_20250316005400210640",
        ],
        "resolution_action": "multi_qrels_added",
        "resolution_note": "서울청년문화패스와 서울청년문화패스 지원 두 문서가 같은 지원 내용을 답하므로 둘 다 정답으로 인정.",
    },
    "oy_eval_repair_q0038": {
        "ground_truth_document_ids": [
            "ontong_youth_20250121005400110344",
            "ontong_youth_20250403005400210687",
        ],
        "resolution_action": "multi_qrels_added",
        "resolution_note": "2025년 전국민마음투자지원사업과 전국민 마음투자 지원사업이 같은 심리상담 바우처 정책을 답하므로 둘 다 정답으로 인정.",
    },
    "oy_eval_repair_q0039": {
        "query": "전남에서 청년 예술인이 창작활동비 지원을 신청하려면 어떤 방식으로 신청하나요?",
        "ground_truth_document_ids": ["ontong_youth_20250119005400210339"],
        "resolution_action": "question_refined",
        "resolution_note": "정답 문서의 실제 주소 요건이 전라남도라 '광주'를 '전남'으로 수정.",
    },
    "oy_eval_repair_q0077": {
        "query": "광주에서 평생교육이용권을 신청하려는 장애인은 어떤 방식으로 신청하나요?",
        "source_document_id": "ontong_youth_20250710005400211177",
        "ground_truth_document_ids": ["ontong_youth_20250710005400211177"],
        "answer_source_field": "plcyAplyMthdCn",
        "resolution_action": "answer_document_reassigned",
        "resolution_note": "제목과 신청방법에 장애인 유형이 명시된 '평생교육이용권[일반·장애인] 지원' 문서로 정답 문서를 재지정.",
    },
    "oy_eval_repair_q0080": {
        "ground_truth_document_ids": [
            "ontong_youth_20260408005400212626",
            "ontong_youth_20260722005400213296",
        ],
        "resolution_action": "multi_qrels_added",
        "resolution_note": "광주 남구 청년 자격증 취득 지원 동일 제목/동일 내용 문서 2건을 모두 정답으로 인정.",
    },
}


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_docs():
    docs = {}
    for line in DOCS_PATH.open("r", encoding="utf-8"):
        if not line.strip():
            continue
        doc = json.loads(line)
        docs[doc["document_id"]] = doc
    return docs


def doc_answer(doc, field_name):
    raw = doc.get("raw_record", {})
    value = raw.get(field_name) or ""
    if value.strip():
        return value.strip()
    fallback = {
        "plcySprtCn": "지원 내용:",
        "addAplyQlfcCndCn": "추가 신청 자격:",
        "plcyAplyMthdCn": "신청 방법:",
        "aplyYmd": "신청 기간:",
        "sbmsnDcmntCn": "제출 서류:",
    }
    marker = fallback.get(field_name)
    text = doc.get("retrieval_text", "")
    if not marker or marker not in text:
        return ""
    after = text.split(marker, 1)[1].strip()
    return after.split("\n\n", 1)[0].strip()


def normalize_doc_ids(value):
    if isinstance(value, list):
        return value
    value = (value or "").strip()
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return [value]


def make_eval_row(
    *,
    query_id,
    query,
    source_query_id,
    source_document_id,
    ground_truth_document_ids,
    source_title,
    source_url,
    primary_category,
    primary_mclsf,
    question_type,
    answer_source_field,
    reference_answer,
    review_source,
    review_note,
    resolution_action="accepted_without_change",
    resolution_note="",
):
    return {
        "query_id": query_id,
        "query": query,
        "source_query_id": source_query_id,
        "source_document_id": source_document_id,
        "ground_truth_document_ids": ground_truth_document_ids,
        "source_title": source_title,
        "source_url": source_url,
        "primary_category": primary_category,
        "primary_mclsf": primary_mclsf,
        "question_type": question_type,
        "answer_source_field": answer_source_field,
        "reference_answer": reference_answer,
        "review_source": review_source,
        "review_label": "1",
        "review_note": review_note,
        "resolution_action": resolution_action,
        "resolution_note": resolution_note,
    }


def main():
    docs = load_docs()
    eval_rows = []
    resolved_rows = []

    adjudicated_rows = read_csv(ADJUDICATED_PATH)
    for row in adjudicated_rows:
        if (row.get("human_review_label") or "").strip() != "1":
            continue
        doc_id = row["source_document_id"]
        eval_rows.append(
            make_eval_row(
                query_id=row["query_id"],
                query=row["query"],
                source_query_id=row["query_id"],
                source_document_id=doc_id,
                ground_truth_document_ids=[doc_id],
                source_title=row["source_title"],
                source_url=row.get("source_url", ""),
                primary_category=row.get("primary_category", ""),
                primary_mclsf=row.get("primary_mclsf", ""),
                question_type=row.get("question_type", ""),
                answer_source_field=row.get("answer_source_field", ""),
                reference_answer=row.get("reference_answer", ""),
                review_source="v2_adjudicated_original_label1",
                review_note=row.get("human_review_note", ""),
            )
        )

    repair_rows = read_csv(REPAIR_LABELED_PATH)
    for row in repair_rows:
        label = (row.get("human_review_label") or "").strip()
        if label not in {"1", "2"}:
            continue

        repair_id = row["repair_query_id"]
        action = "repair_label1_accepted"
        note = ""
        source_doc_id = row["source_document_id"]
        gt_doc_ids = normalize_doc_ids(row.get("ground_truth_document_ids", ""))
        query = row["repaired_query"]
        answer_source_field = row.get("answer_source_field", "")
        reference_answer = row.get("reference_answer", "")

        if label == "2":
            resolution = RESOLVED_LABEL2.get(repair_id)
            if not resolution:
                continue
            action = resolution["resolution_action"]
            note = resolution["resolution_note"]
            query = resolution.get("query", query)
            source_doc_id = resolution.get("source_document_id", source_doc_id)
            gt_doc_ids = resolution["ground_truth_document_ids"]
            answer_source_field = resolution.get("answer_source_field", answer_source_field)
            if source_doc_id in docs:
                source_doc = docs[source_doc_id]
                row["source_title"] = source_doc.get("title", row.get("source_title", ""))
                row["source_url"] = source_doc.get("url", row.get("source_url", ""))
                if resolution.get("answer_source_field"):
                    reference_answer = doc_answer(source_doc, answer_source_field) or source_doc.get("retrieval_text", "")[:500]

            resolved_rows.append(
                {
                    "repair_query_id": repair_id,
                    "source_query_id": row.get("source_query_id", ""),
                    "original_repaired_query": row.get("repaired_query", ""),
                    "final_query": query,
                    "resolution_action": action,
                    "resolution_note": note,
                    "source_document_id": source_doc_id,
                    "ground_truth_document_ids": json.dumps(gt_doc_ids, ensure_ascii=False),
                    "original_review_note": row.get("human_review_note", ""),
                }
            )

        eval_rows.append(
            make_eval_row(
                query_id=repair_id,
                query=query,
                source_query_id=row.get("source_query_id", repair_id),
                source_document_id=source_doc_id,
                ground_truth_document_ids=gt_doc_ids,
                source_title=row.get("source_title", ""),
                source_url=row.get("source_url", ""),
                primary_category=row.get("primary_category", ""),
                primary_mclsf=row.get("primary_mclsf", ""),
                question_type=row.get("question_type", ""),
                answer_source_field=answer_source_field,
                reference_answer=reference_answer,
                review_source="repair_claude_label1" if label == "1" else "repair_label2_resolved",
                review_note=row.get("human_review_note", ""),
                resolution_action=action,
                resolution_note=note,
            )
        )

    qrels_rows = []
    for row in eval_rows:
        for doc_id in row["ground_truth_document_ids"]:
            qrels_rows.append(
                {
                    "query_id": row["query_id"],
                    "document_id": doc_id,
                    "relevance": 1,
                    "source": row["review_source"],
                    "resolution_action": row["resolution_action"],
                }
            )

    eval_rows.sort(key=lambda r: r["query_id"])
    qrels_rows.sort(key=lambda r: (r["query_id"], r["document_id"]))

    # Validate before writing.
    query_ids = [r["query_id"] for r in eval_rows]
    missing_docs = sorted(
        {
            doc_id
            for r in eval_rows
            for doc_id in r["ground_truth_document_ids"]
            if doc_id not in docs
        }
    )
    if len(query_ids) != len(set(query_ids)):
        raise RuntimeError("duplicate query_id found")
    if missing_docs:
        raise RuntimeError(f"missing document ids: {missing_docs}")
    if any(not r["query"].strip() for r in eval_rows):
        raise RuntimeError("blank query found")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eval_json_rows = []
    for row in eval_rows:
        json_row = dict(row)
        eval_json_rows.append(json_row)

    eval_csv_rows = []
    for row in eval_rows:
        csv_row = dict(row)
        csv_row["ground_truth_document_ids"] = json.dumps(
            row["ground_truth_document_ids"], ensure_ascii=False
        )
        eval_csv_rows.append(csv_row)

    eval_fields = [
        "query_id",
        "query",
        "source_query_id",
        "source_document_id",
        "ground_truth_document_ids",
        "source_title",
        "source_url",
        "primary_category",
        "primary_mclsf",
        "question_type",
        "answer_source_field",
        "reference_answer",
        "review_source",
        "review_label",
        "review_note",
        "resolution_action",
        "resolution_note",
    ]
    qrels_fields = ["query_id", "document_id", "relevance", "source", "resolution_action"]

    write_jsonl(OUT_DIR / "eval_questions.jsonl", eval_json_rows)
    write_csv(OUT_DIR / "eval_questions.csv", eval_csv_rows, eval_fields)
    write_jsonl(OUT_DIR / "qrels.jsonl", qrels_rows)
    write_csv(OUT_DIR / "qrels.csv", qrels_rows, qrels_fields)

    resolved_fields = [
        "repair_query_id",
        "source_query_id",
        "original_repaired_query",
        "final_query",
        "resolution_action",
        "resolution_note",
        "source_document_id",
        "ground_truth_document_ids",
        "original_review_note",
    ]
    write_csv(OUT_DIR / "resolved_label2_actions.csv", resolved_rows, resolved_fields)

    manifest = {
        "dataset_name": "Ontong Youth MVP Evaluation Questions",
        "dataset_version": "ontong_youth_eval_questions_v2_final_47",
        "created_at": datetime.now(KST).isoformat(),
        "inputs": {
            "documents": str(DOCS_PATH),
            "adjudicated_standardized": str(ADJUDICATED_PATH),
            "repair_claude_labeled": str(REPAIR_LABELED_PATH),
        },
        "counts": {
            "eval_questions": len(eval_rows),
            "qrels_rows": len(qrels_rows),
            "original_v2_label1": sum(
                1 for r in eval_rows if r["review_source"] == "v2_adjudicated_original_label1"
            ),
            "repair_label1": sum(
                1 for r in eval_rows if r["review_source"] == "repair_claude_label1"
            ),
            "repair_label2_resolved": sum(
                1 for r in eval_rows if r["review_source"] == "repair_label2_resolved"
            ),
            "multi_qrels_queries": sum(
                1 for r in eval_rows if len(r["ground_truth_document_ids"]) > 1
            ),
        },
        "distributions": {
            "question_type": dict(Counter(r["question_type"] for r in eval_rows)),
            "primary_category": dict(Counter(r["primary_category"] for r in eval_rows)),
            "resolution_action": dict(Counter(r["resolution_action"] for r in eval_rows)),
        },
        "validation": {
            "duplicate_query_id": len(query_ids) - len(set(query_ids)),
            "blank_query": sum(1 for r in eval_rows if not r["query"].strip()),
            "missing_ground_truth_document_ids": missing_docs,
            "blank_reference_answer": sum(
                1 for r in eval_rows if not r.get("reference_answer", "").strip()
            ),
        },
        "notes": [
            "This creates evaluation questions and document-level qrels only.",
            "No chunking, embedding, Chroma indexing, or retrieval execution is performed.",
            "ground_truth_chunk_ids must be derived later after chunking.",
        ],
        "output_files": {
            "eval_questions_jsonl": str(OUT_DIR / "eval_questions.jsonl"),
            "eval_questions_csv": str(OUT_DIR / "eval_questions.csv"),
            "qrels_jsonl": str(OUT_DIR / "qrels.jsonl"),
            "qrels_csv": str(OUT_DIR / "qrels.csv"),
            "resolved_label2_actions_csv": str(OUT_DIR / "resolved_label2_actions.csv"),
            "manifest": str(OUT_DIR / "eval_questions_manifest.json"),
        },
    }
    (OUT_DIR / "eval_questions_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

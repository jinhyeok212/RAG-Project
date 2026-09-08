from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(r"C:\말똥가리\data\processed\ontong_youth_mvp_400\ontong_youth_mvp_400_catalog.jsonl")
DEFAULT_OUTPUT_DIR = Path(r"C:\말똥가리\data\processed\ontong_youth_mvp_400_documents")


TEXT_FIELDS = [
    ("정책명", "plcyNm"),
    ("정책 요약", "plcyExplnCn"),
    ("지원 내용", "plcySprtCn"),
    ("참여 대상", "ptcpPrpTrgtCn"),
    ("추가 신청 자격", "addAplyQlfcCndCn"),
    ("신청 기간", "aplyYmd"),
    ("사업 기간", "bizPrdEtcCn"),
    ("신청 방법", "plcyAplyMthdCn"),
    ("심사 방법", "srngMthdCn"),
    ("제출 서류", "sbmsnDcmntCn"),
    ("기타 사항", "etcMttrCn"),
    ("주관 기관", "sprvsnInstCdNm"),
    ("운영 기관", "operInstCdNm"),
    ("정책 대분류", "primary_category"),
    ("정책 중분류", "primary_mclsf"),
    ("정책 키워드", "plcyKywdNm"),
    ("신청 URL", "aplyUrlAddr"),
    ("참고 URL", "refUrlAddr1"),
]


RAW_API_FIELDS = [
    "plcyNo",
    "bscPlanCycl",
    "bscPlanPlcyWayNo",
    "bscPlanFcsAsmtNo",
    "bscPlanAsmtNo",
    "pvsnInstGroupCd",
    "plcyPvsnMthdCd",
    "plcyAprvSttsCd",
    "plcyNm",
    "plcyKywdNm",
    "plcyExplnCn",
    "lclsfNm",
    "mclsfNm",
    "plcySprtCn",
    "sprvsnInstCd",
    "sprvsnInstCdNm",
    "sprvsnInstPicNm",
    "operInstCd",
    "operInstCdNm",
    "operInstPicNm",
    "sprtSclLmtYn",
    "aplyPrdSeCd",
    "bizPrdSeCd",
    "bizPrdBgngYmd",
    "bizPrdEndYmd",
    "bizPrdEtcCn",
    "plcyAplyMthdCn",
    "srngMthdCn",
    "aplyUrlAddr",
    "sbmsnDcmntCn",
    "etcMttrCn",
    "refUrlAddr1",
    "refUrlAddr2",
    "sprtSclCnt",
    "sprtArvlSeqYn",
    "sprtTrgtMinAge",
    "sprtTrgtMaxAge",
    "sprtTrgtAgeLmtYn",
    "mrgSttsCd",
    "earnCndSeCd",
    "earnMinAmt",
    "earnMaxAmt",
    "earnEtcCn",
    "addAplyQlfcCndCn",
    "ptcpPrpTrgtCn",
    "inqCnt",
    "rgtrInstCd",
    "rgtrInstCdNm",
    "rgtrUpInstCd",
    "rgtrUpInstCdNm",
    "rgtrHghrkInstCd",
    "rgtrHghrkInstCdNm",
    "zipCd",
    "plcyMajorCd",
    "jobCd",
    "schoolCd",
    "aplyYmd",
    "frstRegDt",
    "lastMdfcnDt",
    "sbizCd",
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


def clean_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
        if isinstance(loaded, list):
            return [str(item) for item in loaded]
    except json.JSONDecodeError:
        pass
    return [item.strip() for item in text.split(",") if item.strip()]


def make_structured_text(row: dict[str, Any]) -> str:
    lines = []
    for label, field in TEXT_FIELDS:
        value = clean_text(row.get(field, ""))
        if value:
            lines.append(f"{label}: {value}")
    return "\n\n".join(lines).strip()


def make_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "ontong_youth",
        "source_dataset": "온통청년 청년정책 API",
        "source_policy_no": row.get("plcyNo", ""),
        "primary_category": row.get("primary_category", ""),
        "primary_mclsf": row.get("primary_mclsf", ""),
        "normalized_lclsf_list": parse_json_list(row.get("normalized_lclsf_list")),
        "normalized_mclsf_list": parse_json_list(row.get("normalized_mclsf_list")),
        "normalized_keywords": parse_json_list(row.get("normalized_keywords")),
        "raw_lclsfNm": row.get("lclsfNm", ""),
        "raw_mclsfNm": row.get("mclsfNm", ""),
        "raw_plcyKywdNm": row.get("plcyKywdNm", ""),
        "supervising_agency": row.get("sprvsnInstCdNm", ""),
        "operating_agency": row.get("operInstCdNm", ""),
        "application_period": row.get("aplyYmd", ""),
        "business_start_date": row.get("bizPrdBgngYmd", ""),
        "business_end_date": row.get("bizPrdEndYmd", ""),
        "business_period_note": clean_text(row.get("bizPrdEtcCn", "")),
        "target_min_age": row.get("sprtTrgtMinAge", ""),
        "target_max_age": row.get("sprtTrgtMaxAge", ""),
        "target_age_limit_yn": row.get("sprtTrgtAgeLmtYn", ""),
        "income_condition_code": row.get("earnCndSeCd", ""),
        "income_min_amount": row.get("earnMinAmt", ""),
        "income_max_amount": row.get("earnMaxAmt", ""),
        "marriage_status_code": row.get("mrgSttsCd", ""),
        "region_zip_codes": row.get("zipCd", ""),
        "application_url": row.get("aplyUrlAddr", ""),
        "reference_url_1": row.get("refUrlAddr1", ""),
        "reference_url_2": row.get("refUrlAddr2", ""),
        "first_registered_at": row.get("frstRegDt", ""),
        "last_modified_at": row.get("lastMdfcnDt", ""),
        "document_quality_score": int(row.get("document_quality_score") or 0),
        "document_quality_reasons": parse_json_list(row.get("document_quality_reasons")),
        "normalization_status": row.get("normalization_status", ""),
        "mvp_selection_version": "ontong_youth_mvp_400_selection_v1",
    }


def make_raw_record(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in RAW_API_FIELDS}


def make_document(row: dict[str, Any]) -> dict[str, Any]:
    content = make_structured_text(row)
    retrieval_text = content
    metadata = make_metadata(row)
    return {
        "document_id": row.get("document_id", ""),
        "title": clean_text(row.get("plcyNm", "")),
        "source": "ontong_youth",
        "url": row.get("aplyUrlAddr") or row.get("refUrlAddr1") or row.get("refUrlAddr2") or "",
        "content": content,
        "retrieval_text": retrieval_text,
        "metadata": metadata,
        "raw_record": make_raw_record(row),
    }


def run(input_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(input_path)
    documents = [make_document(row) for row in rows]

    doc_ids = [doc["document_id"] for doc in documents]
    categories = Counter(doc["metadata"]["primary_category"] for doc in documents)
    middle = Counter(doc["metadata"]["primary_mclsf"] for doc in documents)
    lengths = [len(doc["retrieval_text"]) for doc in documents]
    quality_rows = [
        {"metric": "document_count", "value": len(documents)},
        {"metric": "duplicate_document_id_count", "value": len(doc_ids) - len(set(doc_ids))},
        {"metric": "blank_document_id_count", "value": sum(1 for value in doc_ids if not value)},
        {"metric": "blank_title_count", "value": sum(1 for doc in documents if not doc["title"])},
        {"metric": "blank_retrieval_text_count", "value": sum(1 for doc in documents if not doc["retrieval_text"])},
        {"metric": "retrieval_text_min_length", "value": min(lengths) if lengths else 0},
        {"metric": "retrieval_text_avg_length", "value": round(sum(lengths) / len(lengths), 3) if lengths else 0},
        {"metric": "retrieval_text_max_length", "value": max(lengths) if lengths else 0},
    ]
    for category, count in categories.most_common():
        quality_rows.append({"metric": f"primary_category.{category}", "value": count})
    for item, count in middle.most_common():
        quality_rows.append({"metric": f"primary_mclsf.{item}", "value": count})

    schema = {
        "schema_name": "ontong_youth_rag_document_v1",
        "description": "온통청년 MVP 400 정책을 RAG corpus로 사용하기 위한 표준 문서 JSONL 스키마",
        "record_unit": "one policy = one document",
        "chunking_target_field": "retrieval_text",
        "fields": {
            "document_id": "문서 고유 ID. ontong_youth_{plcyNo}",
            "title": "정책명. 원본 plcyNm에서 생성",
            "source": "출처 식별자. ontong_youth",
            "url": "대표 URL. 신청 URL 우선, 없으면 참고 URL",
            "content": "정책 주요 필드를 사람이 읽기 좋은 라벨형 텍스트로 결합한 전체 본문",
            "retrieval_text": "3번 담당자가 실제 chunking/search에 사용할 텍스트. 현재 content와 동일",
            "metadata": "분류, 기관, 신청기간, 조건, URL, 품질점수 등 검색/필터링용 메타데이터",
            "raw_record": "온통청년 API 원본 주요 필드 보존",
        },
        "metadata_fields": sorted(documents[0]["metadata"].keys()) if documents else [],
        "raw_record_fields": RAW_API_FIELDS,
        "notes": [
            "원본 API 필드는 raw_record에 보존한다.",
            "검색/청킹은 content가 아니라 retrieval_text를 사용한다.",
            "아직 chunk_id, embedding, vector DB, 평가 질문은 생성하지 않는다.",
        ],
    }
    manifest = {
        "dataset_name": "Ontong Youth MVP 400 RAG Documents",
        "dataset_version": "ontong_youth_mvp_400_documents_v1",
        "source_mvp_catalog": str(input_path),
        "schema_name": schema["schema_name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(documents),
        "quality": {row["metric"]: row["value"] for row in quality_rows},
        "output_files": {
            "documents_jsonl": str(output_dir / "documents.jsonl"),
            "documents_sample_json": str(output_dir / "documents_sample.json"),
            "document_schema_json": str(output_dir / "document_schema.json"),
            "document_schema_md": str(output_dir / "DOCUMENT_SCHEMA.md"),
            "document_quality_report_csv": str(output_dir / "document_quality_report.csv"),
            "document_manifest_json": str(output_dir / "document_manifest.json"),
        },
    }

    schema_md = f"""# Ontong Youth RAG Document Schema v1

## 목적

온통청년 MVP 400개 정책을 RAG 파이프라인에 넣기 위한 공통 문서 구조다. 정책 1개가 문서 1개가 된다.

## 레코드 단위

`one policy = one document`

## 3번 Chunking 담당자가 사용할 필드

`retrieval_text`

`content`와 `raw_record`는 보존/검수용이며, 실제 청킹 대상은 `retrieval_text`다.

## 최상위 필드

- `document_id`: 문서 고유 ID. `ontong_youth_{{plcyNo}}`
- `title`: 정책명
- `source`: `ontong_youth`
- `url`: 대표 URL
- `content`: 주요 정책 필드를 라벨형 텍스트로 결합한 본문
- `retrieval_text`: 검색/청킹 대상 텍스트
- `metadata`: 분류, 기관, 기간, 조건, URL, 품질점수
- `raw_record`: 온통청년 API 원본 주요 필드

## 아직 하지 않은 작업

- 평가 질문 생성
- `ground_truth_document_ids` 지정
- Chunking
- Chunk ID 생성
- Embedding
- Chroma 저장
"""

    write_jsonl(output_dir / "documents.jsonl", documents)
    write_json(output_dir / "documents_sample.json", documents[:5])
    write_json(output_dir / "document_schema.json", schema)
    (output_dir / "DOCUMENT_SCHEMA.md").write_text(schema_md, encoding="utf-8")
    write_csv(output_dir / "document_quality_report.csv", quality_rows, ["metric", "value"])
    write_json(output_dir / "document_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = run(args.input, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

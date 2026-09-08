import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from chunk_c2_section_800 import (
    METADATA_KEEP,
    build_chroma_ready_chunk,
    build_chunks,
    clean_value,
    flatten_metadata_value,
    percentile,
    split_body,
    validate_chroma_metadata,
    validate_chunks,
    write_jsonl,
    write_quality_report,
)


INPUT_PATH = Path("01.document/documents.jsonl")
BASE_OUTPUT_DIR = Path("05.chunks")
MAX_CHARS = 800
OVERLAP_CHARS = 120


ATOMIC_FIELD_DEFS = [
    ("overview", "policy_title", ("doc", "title")),
    ("overview", "policy_summary", ("raw", "plcyExplnCn")),
    ("overview", "primary_category", ("meta", "primary_category")),
    ("overview", "primary_mclsf", ("meta", "primary_mclsf")),
    ("overview", "keywords", ("meta", "normalized_keywords")),
    ("overview", "supervising_agency", ("meta", "supervising_agency")),
    ("overview", "operating_agency", ("meta", "operating_agency")),
    ("support_content", "support_content", ("raw", "plcySprtCn")),
    ("eligibility", "participation_target", ("raw", "ptcpPrpTrgtCn")),
    ("eligibility", "additional_eligibility", ("raw", "addAplyQlfcCndCn")),
    ("eligibility", "income_note", ("raw", "earnEtcCn")),
    ("eligibility", "target_age", ("derived", "target_age")),
    ("eligibility", "income_condition", ("derived", "income_condition")),
    ("eligibility", "marriage_status_code", ("meta", "marriage_status_code")),
    ("eligibility", "region_zip_codes", ("meta", "region_zip_codes")),
    ("application_period", "application_period", ("meta", "application_period")),
    ("application_period", "business_period", ("derived", "business_period")),
    ("application_method", "application_method", ("raw", "plcyAplyMthdCn")),
    ("application_method", "application_url", ("meta", "application_url")),
    ("screening", "screening_method", ("raw", "srngMthdCn")),
    ("required_documents", "required_documents", ("raw", "sbmsnDcmntCn")),
    ("notes", "notes", ("raw", "etcMttrCn")),
    ("notes", "reference_url_1", ("meta", "reference_url_1")),
    ("notes", "reference_url_2", ("meta", "reference_url_2")),
    ("notes", "source_url", ("doc", "url")),
]


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_value(doc, source):
    source_type, key = source
    metadata = doc.get("metadata") or {}
    raw = doc.get("raw_record") or {}

    if source_type == "doc":
        return clean_value(doc.get(key))
    if source_type == "meta":
        return clean_value(metadata.get(key))
    if source_type == "raw":
        return clean_value(raw.get(key))
    if source_type == "derived" and key == "target_age":
        parts = [
            f"target_min_age: {clean_value(metadata.get('target_min_age'))}",
            f"target_max_age: {clean_value(metadata.get('target_max_age'))}",
            f"target_age_limit_yn: {clean_value(metadata.get('target_age_limit_yn'))}",
        ]
        return "\n".join(part for part in parts if not part.endswith(": "))
    if source_type == "derived" and key == "income_condition":
        parts = [
            f"income_condition_code: {clean_value(metadata.get('income_condition_code'))}",
            f"income_min_amount: {clean_value(metadata.get('income_min_amount'))}",
            f"income_max_amount: {clean_value(metadata.get('income_max_amount'))}",
        ]
        return "\n".join(part for part in parts if not part.endswith(": "))
    if source_type == "derived" and key == "business_period":
        parts = [
            f"business_start_date: {clean_value(metadata.get('business_start_date'))}",
            f"business_end_date: {clean_value(metadata.get('business_end_date'))}",
            f"business_period_note: {clean_value(metadata.get('business_period_note'))}",
        ]
        return "\n".join(part for part in parts if not part.endswith(": "))
    raise ValueError(f"Unknown source: {source}")


def build_atomic_header(doc, section, field_name):
    metadata = doc.get("metadata") or {}
    header_fields = [
        ("policy_title", clean_value(doc.get("title"))),
        ("document_id", clean_value(doc.get("document_id"))),
        ("primary_category", clean_value(metadata.get("primary_category"))),
        ("primary_mclsf", clean_value(metadata.get("primary_mclsf"))),
        ("keywords", clean_value(metadata.get("normalized_keywords"))),
        ("agency", clean_value(metadata.get("supervising_agency"))),
        ("section", section),
        ("field", field_name),
    ]
    return "\n".join(f"{label}: {value}" for label, value in header_fields if value)


def metadata_for_doc(doc):
    metadata = doc.get("metadata") or {}
    return {key: metadata.get(key) for key in METADATA_KEEP if key in metadata}


def make_atomic_chunk(doc, section, field_name, part_index, chunk_index, body_part):
    document_id = clean_value(doc.get("document_id"))
    header = build_atomic_header(doc, section, field_name)
    text = f"{header}\n\n{field_name}: {body_part}".strip()
    return {
        "chunk_id": f"{document_id}__{section}__{field_name}__{part_index:03d}",
        "document_id": document_id,
        "chunk_index": chunk_index,
        "section": section,
        "field": field_name,
        "section_chunk_index": part_index,
        "text": text,
        "body_text": body_part,
        "text_char_count": len(text),
        "body_char_count": len(body_part),
        "source": doc.get("source"),
        "title": doc.get("title"),
        "url": doc.get("url"),
        "metadata": metadata_for_doc(doc),
    }


def build_atomic_chunks(docs):
    chunks = []
    doc_reports = []

    for doc in docs:
        document_id = clean_value(doc.get("document_id"))
        doc_chunks = []
        seen_sections = []
        blank_fields = []
        chunk_index = 0

        for section, field_name, source in ATOMIC_FIELD_DEFS:
            value = get_value(doc, source)
            if not value:
                blank_fields.append(field_name)
                continue

            parts = split_body(value, max_chars=MAX_CHARS, overlap_chars=OVERLAP_CHARS)
            for part_index, body_part in enumerate(parts):
                chunk = make_atomic_chunk(
                    doc=doc,
                    section=section,
                    field_name=field_name,
                    part_index=part_index,
                    chunk_index=chunk_index,
                    body_part=body_part,
                )
                chunks.append(chunk)
                doc_chunks.append(chunk)
                seen_sections.append(section)
                chunk_index += 1

        doc_reports.append(
            {
                "document_id": document_id,
                "title": clean_value(doc.get("title")),
                "retrieval_text_char_count": len(clean_value(doc.get("retrieval_text"))),
                "chunk_count": len(doc_chunks),
                "sections": "|".join(sorted(set(seen_sections))),
                "max_body_char_count": max((chunk["body_char_count"] for chunk in doc_chunks), default=0),
                "max_text_char_count": max((chunk["text_char_count"] for chunk in doc_chunks), default=0),
                "blank_fields": "|".join(blank_fields),
            }
        )

    return chunks, doc_reports


def build_parent_documents(docs):
    rows = []
    for doc in docs:
        metadata = {
            key: flatten_metadata_value(value)
            for key, value in metadata_for_doc(doc).items()
        }
        metadata.update(
            {
                "document_id": clean_value(doc.get("document_id")),
                "title": flatten_metadata_value(doc.get("title")),
                "url": flatten_metadata_value(doc.get("url")),
                "source": flatten_metadata_value(doc.get("source")),
            }
        )
        rows.append(
            {
                "document_id": clean_value(doc.get("document_id")),
                "title": doc.get("title"),
                "url": doc.get("url"),
                "text": clean_value(doc.get("retrieval_text")),
                "metadata": metadata,
            }
        )
    return rows


def add_parent_metadata(chroma_rows):
    for row in chroma_rows:
        row["metadata"]["parent_document_id"] = row["metadata"]["document_id"]
        row["metadata"]["retrieval_expansion"] = "parent_document"
    return chroma_rows


def write_atomic_quality_report(path, doc_reports):
    fieldnames = [
        "document_id",
        "title",
        "retrieval_text_char_count",
        "chunk_count",
        "sections",
        "max_body_char_count",
        "max_text_char_count",
        "blank_fields",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(doc_reports)


def build_manifest(
    *,
    experiment_id,
    strategy_name,
    strategy,
    docs,
    chunks,
    chroma_rows,
    doc_reports,
    output_files,
):
    body_lengths = [chunk["body_char_count"] for chunk in chunks]
    text_lengths = [chunk["text_char_count"] for chunk in chunks]
    doc_chunk_counts = [row["chunk_count"] for row in doc_reports]
    expected_doc_ids = {clean_value(doc.get("document_id")) for doc in docs}

    validation = validate_chunks(chunks, expected_doc_ids)
    chroma_invalid = validate_chroma_metadata(chroma_rows)

    return {
        "dataset_name": f"Ontong Youth MVP 400 {strategy_name}",
        "dataset_version": f"{experiment_id}_v1",
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(INPUT_PATH),
        "chunking_strategy": strategy,
        "counts": {
            "document_count": len(docs),
            "chunk_count": len(chunks),
            "avg_chunks_per_document": round(len(chunks) / len(docs), 3) if docs else 0,
            "min_chunks_per_document": min(doc_chunk_counts) if doc_chunk_counts else 0,
            "max_chunks_per_document": max(doc_chunk_counts) if doc_chunk_counts else 0,
            "section_counts": dict(sorted(Counter(chunk["section"] for chunk in chunks).items())),
        },
        "lengths": {
            "body_char_count": {
                "min": min(body_lengths) if body_lengths else 0,
                "avg": round(sum(body_lengths) / len(body_lengths), 1) if body_lengths else 0,
                "p50": percentile(body_lengths, 0.50),
                "p90": percentile(body_lengths, 0.90),
                "p95": percentile(body_lengths, 0.95),
                "max": max(body_lengths) if body_lengths else 0,
            },
            "text_char_count_with_header": {
                "min": min(text_lengths) if text_lengths else 0,
                "avg": round(sum(text_lengths) / len(text_lengths), 1) if text_lengths else 0,
                "p50": percentile(text_lengths, 0.50),
                "p90": percentile(text_lengths, 0.90),
                "p95": percentile(text_lengths, 0.95),
                "max": max(text_lengths) if text_lengths else 0,
            },
        },
        "validation": validation,
        "chroma_ready_validation": {
            "row_count": len(chroma_rows),
            "invalid_metadata_value_examples": chroma_invalid,
            "metadata_is_flat_scalar": not chroma_invalid,
        },
        "output_files": output_files,
    }


def write_experiment_brief(path, *, title, summary, files, details, validation):
    lines = [
        f"# {title}",
        "",
        "## 한 줄 요약",
        "",
        summary,
        "",
        "## 전달 파일",
        "",
    ]
    lines.extend(f"- {label}: `{file_path}`" for label, file_path in files)
    lines.extend(["", "## 방식 설명", ""])
    lines.extend(details)
    lines.extend(["", "## 검증 결과", ""])
    lines.extend(f"- {item}" for item in validation)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_exp_02(docs):
    experiment_id = "exp_02_section_800_parent"
    output_dir = BASE_OUTPUT_DIR / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks, doc_reports = build_chunks(docs, MAX_CHARS, OVERLAP_CHARS)
    chroma_rows = add_parent_metadata([build_chroma_ready_chunk(chunk) for chunk in chunks])
    parent_documents = build_parent_documents(docs)

    chunks_path = output_dir / f"{experiment_id}_chunks.jsonl"
    chroma_path = output_dir / f"{experiment_id}_chunks_chroma_ready.jsonl"
    parents_path = output_dir / f"{experiment_id}_parent_documents.jsonl"
    quality_path = output_dir / f"{experiment_id}_quality_report.csv"
    manifest_path = output_dir / f"{experiment_id}_manifest.json"
    brief_path = output_dir / "EXPERIMENT_BRIEF.md"

    write_jsonl(chunks_path, chunks)
    write_jsonl(chroma_path, chroma_rows)
    write_jsonl(parents_path, parent_documents)
    write_quality_report(quality_path, doc_reports)

    output_files = {
        "chunks_jsonl": str(chunks_path),
        "chunks_chroma_ready_jsonl": str(chroma_path),
        "parent_documents_jsonl": str(parents_path),
        "quality_report_csv": str(quality_path),
        "manifest": str(manifest_path),
        "brief_md": str(brief_path),
    }
    manifest = build_manifest(
        experiment_id=experiment_id,
        strategy_name="EXP 02 Section 800 Parent Chunks",
        strategy={
            "name": "D1 C2 section-based chunks with parent expansion",
            "base_strategy": "C2 section-based chunking",
            "target_field": "retrieval_text",
            "max_body_chars": MAX_CHARS,
            "overlap_chars": OVERLAP_CHARS,
            "header_included": True,
            "parent_expansion": True,
            "parent_document_file": str(parents_path),
        },
        docs=docs,
        chunks=chunks,
        chroma_rows=chroma_rows,
        doc_reports=doc_reports,
        output_files=output_files,
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_experiment_brief(
        brief_path,
        title="EXP 02 Section 800 Parent 청킹 설명",
        summary="C2 섹션 기반 청킹 결과에 parent document 확장용 파일을 함께 제공하는 후보입니다.",
        files=[
            ("임베딩용 파일", chroma_path),
            ("parent 확장용 문서 파일", parents_path),
            ("매니페스트", manifest_path),
            ("품질 리포트", quality_path),
            ("원본형 청크", chunks_path),
        ],
        details=[
            "- C2와 같은 섹션 기반 청킹을 사용합니다.",
            "- 청크 검색은 `text` 기준으로 수행하고, 검색 후 `metadata.parent_document_id`로 원문 정책 문서를 확장할 수 있게 했습니다.",
            "- parent 확장용 전체 문서는 `parent_documents.jsonl`의 `text` 필드에 들어 있습니다.",
            "- 목적은 세부 항목 검색 정확도와 답변 생성 시 전체 정책 맥락을 함께 가져가는 것입니다.",
        ],
        validation=[
            f"문서 수: {manifest['counts']['document_count']}",
            f"청크 수: {manifest['counts']['chunk_count']}",
            f"문서당 평균 청크 수: {manifest['counts']['avg_chunks_per_document']}",
            f"빈 청크: {manifest['validation']['blank_chunk_count']}",
            f"중복 chunk_id: {manifest['validation']['duplicate_chunk_id_count']}",
            f"누락 문서: {len(manifest['validation']['missing_document_ids'])}",
            f"Chroma metadata 타입 오류: {len(manifest['chroma_ready_validation']['invalid_metadata_value_examples'])}",
        ],
    )
    return manifest


def run_exp_03(docs):
    experiment_id = "exp_03_field_atomic_parent"
    output_dir = BASE_OUTPUT_DIR / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks, doc_reports = build_atomic_chunks(docs)
    chroma_rows = add_parent_metadata([build_chroma_ready_chunk(chunk) for chunk in chunks])
    for row, chunk in zip(chroma_rows, chunks):
        row["metadata"]["field"] = chunk["field"]
    parent_documents = build_parent_documents(docs)

    chunks_path = output_dir / f"{experiment_id}_chunks.jsonl"
    chroma_path = output_dir / f"{experiment_id}_chunks_chroma_ready.jsonl"
    parents_path = output_dir / f"{experiment_id}_parent_documents.jsonl"
    quality_path = output_dir / f"{experiment_id}_quality_report.csv"
    manifest_path = output_dir / f"{experiment_id}_manifest.json"
    brief_path = output_dir / "EXPERIMENT_BRIEF.md"

    write_jsonl(chunks_path, chunks)
    write_jsonl(chroma_path, chroma_rows)
    write_jsonl(parents_path, parent_documents)
    write_atomic_quality_report(quality_path, doc_reports)

    output_files = {
        "chunks_jsonl": str(chunks_path),
        "chunks_chroma_ready_jsonl": str(chroma_path),
        "parent_documents_jsonl": str(parents_path),
        "quality_report_csv": str(quality_path),
        "manifest": str(manifest_path),
        "brief_md": str(brief_path),
    }
    manifest = build_manifest(
        experiment_id=experiment_id,
        strategy_name="EXP 03 Field Atomic Parent Chunks",
        strategy={
            "name": "E1 field atomic chunks with parent expansion",
            "target_field": "raw_record plus metadata atomic fields",
            "max_body_chars": MAX_CHARS,
            "overlap_chars": OVERLAP_CHARS,
            "header_included": True,
            "field_atomic": True,
            "parent_expansion": True,
            "parent_document_file": str(parents_path),
        },
        docs=docs,
        chunks=chunks,
        chroma_rows=chroma_rows,
        doc_reports=doc_reports,
        output_files=output_files,
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_experiment_brief(
        brief_path,
        title="EXP 03 Field Atomic Parent 청킹 설명",
        summary="정책 문서를 필드 단위로 더 잘게 나누고 parent document 확장용 파일을 함께 제공하는 후보입니다.",
        files=[
            ("임베딩용 파일", chroma_path),
            ("parent 확장용 문서 파일", parents_path),
            ("매니페스트", manifest_path),
            ("품질 리포트", quality_path),
            ("원본형 청크", chunks_path),
        ],
        details=[
            "- 지원내용, 참여대상, 추가 자격, 신청기간, 신청방법, 제출서류 같은 필드를 각각 atomic chunk로 만듭니다.",
            "- 각 청크에는 `section`과 `field` 태그가 같이 들어갑니다.",
            "- 신청기간, 신청방법, 제출서류, 자격조건처럼 답변 위치가 명확한 질문에 강한지 확인하기 위한 후보입니다.",
            "- 검색 결과가 너무 세부적일 수 있으므로 `metadata.parent_document_id`로 전체 정책 문서를 확장하는 전제를 둡니다.",
        ],
        validation=[
            f"문서 수: {manifest['counts']['document_count']}",
            f"청크 수: {manifest['counts']['chunk_count']}",
            f"문서당 평균 청크 수: {manifest['counts']['avg_chunks_per_document']}",
            f"빈 청크: {manifest['validation']['blank_chunk_count']}",
            f"중복 chunk_id: {manifest['validation']['duplicate_chunk_id_count']}",
            f"누락 문서: {len(manifest['validation']['missing_document_ids'])}",
            f"Chroma metadata 타입 오류: {len(manifest['chroma_ready_validation']['invalid_metadata_value_examples'])}",
        ],
    )
    return manifest


def main():
    docs = read_jsonl(INPUT_PATH)
    manifests = [run_exp_02(docs), run_exp_03(docs)]
    print(json.dumps({"experiments": manifests}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

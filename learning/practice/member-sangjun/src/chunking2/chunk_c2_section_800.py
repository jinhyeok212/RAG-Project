import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SECTION_DEFS = [
    (
        "overview",
        [
            ("policy_title", ("doc", "title")),
            ("policy_summary", ("raw", "plcyExplnCn")),
            ("primary_category", ("meta", "primary_category")),
            ("primary_mclsf", ("meta", "primary_mclsf")),
            ("keywords", ("meta", "normalized_keywords")),
            ("supervising_agency", ("meta", "supervising_agency")),
            ("operating_agency", ("meta", "operating_agency")),
        ],
    ),
    (
        "support_content",
        [
            ("support_content", ("raw", "plcySprtCn")),
        ],
    ),
    (
        "eligibility",
        [
            ("participation_target", ("raw", "ptcpPrpTrgtCn")),
            ("additional_eligibility", ("raw", "addAplyQlfcCndCn")),
            ("income_note", ("raw", "earnEtcCn")),
            ("target_min_age", ("meta", "target_min_age")),
            ("target_max_age", ("meta", "target_max_age")),
            ("target_age_limit_yn", ("meta", "target_age_limit_yn")),
            ("income_condition_code", ("meta", "income_condition_code")),
            ("income_min_amount", ("meta", "income_min_amount")),
            ("income_max_amount", ("meta", "income_max_amount")),
            ("marriage_status_code", ("meta", "marriage_status_code")),
            ("region_zip_codes", ("meta", "region_zip_codes")),
        ],
    ),
    (
        "application",
        [
            ("application_period", ("meta", "application_period")),
            ("business_start_date", ("meta", "business_start_date")),
            ("business_end_date", ("meta", "business_end_date")),
            ("business_period_note", ("meta", "business_period_note")),
            ("application_method", ("raw", "plcyAplyMthdCn")),
            ("application_url", ("meta", "application_url")),
        ],
    ),
    (
        "screening",
        [
            ("screening_method", ("raw", "srngMthdCn")),
        ],
    ),
    (
        "required_documents",
        [
            ("required_documents", ("raw", "sbmsnDcmntCn")),
        ],
    ),
    (
        "notes",
        [
            ("notes", ("raw", "etcMttrCn")),
            ("reference_url_1", ("meta", "reference_url_1")),
            ("reference_url_2", ("meta", "reference_url_2")),
            ("source_url", ("doc", "url")),
        ],
    ),
]


METADATA_KEEP = [
    "source",
    "source_dataset",
    "source_policy_no",
    "primary_category",
    "primary_mclsf",
    "normalized_lclsf_list",
    "normalized_mclsf_list",
    "normalized_keywords",
    "supervising_agency",
    "operating_agency",
    "application_period",
    "business_start_date",
    "business_end_date",
    "business_period_note",
    "target_min_age",
    "target_max_age",
    "target_age_limit_yn",
    "income_condition_code",
    "income_min_amount",
    "income_max_amount",
    "marriage_status_code",
    "region_zip_codes",
    "application_url",
    "reference_url_1",
    "reference_url_2",
    "first_registered_at",
    "last_modified_at",
    "document_quality_score",
    "normalization_status",
    "mvp_selection_version",
]


def clean_value(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_value(doc, source):
    source_type, key = source
    if source_type == "doc":
        return clean_value(doc.get(key))
    if source_type == "meta":
        return clean_value((doc.get("metadata") or {}).get(key))
    if source_type == "raw":
        return clean_value((doc.get("raw_record") or {}).get(key))
    raise ValueError(f"Unknown source type: {source_type}")


def build_section_body(doc, fields):
    lines = []
    for label, source in fields:
        value = get_value(doc, source)
        if value:
            lines.append(f"{label}: {value}")
    return "\n\n".join(lines).strip()


def build_header(doc, section):
    metadata = doc.get("metadata") or {}
    keywords = clean_value(metadata.get("normalized_keywords"))
    header_fields = [
        ("policy_title", clean_value(doc.get("title"))),
        ("document_id", clean_value(doc.get("document_id"))),
        ("primary_category", clean_value(metadata.get("primary_category"))),
        ("primary_mclsf", clean_value(metadata.get("primary_mclsf"))),
        ("keywords", keywords),
        ("agency", clean_value(metadata.get("supervising_agency"))),
        ("section", section),
    ]
    return "\n".join(f"{label}: {value}" for label, value in header_fields if value)


def split_large_unit(unit, max_chars):
    unit = unit.strip()
    if len(unit) <= max_chars:
        return [unit]

    sentence_parts = re.split(r"(?<=[.!?])\s+", unit)
    if len(sentence_parts) > 1:
        parts = []
        current = ""
        for sentence in sentence_parts:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    parts.append(current)
                current = sentence
        if current:
            parts.append(current)
        if all(len(part) <= max_chars for part in parts):
            return parts

    return [unit[i : i + max_chars] for i in range(0, len(unit), max_chars)]


def split_body(body, max_chars, overlap_chars):
    body = body.strip()
    if not body:
        return []
    if len(body) <= max_chars:
        return [body]

    raw_units = re.split(r"\n\s*\n", body)
    units = []
    for raw_unit in raw_units:
        raw_unit = raw_unit.strip()
        if not raw_unit:
            continue
        if len(raw_unit) <= max_chars:
            units.append(raw_unit)
            continue
        line_units = [line.strip() for line in raw_unit.split("\n") if line.strip()]
        if len(line_units) > 1 and all(len(line) <= max_chars for line in line_units):
            units.extend(line_units)
            continue
        for line in line_units or [raw_unit]:
            units.extend(split_large_unit(line, max_chars))

    chunks = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            overlap = current[-overlap_chars:].strip() if overlap_chars else ""
            current = f"{overlap}\n\n{unit}".strip() if overlap else unit
            if len(current) > max_chars:
                chunks.extend(split_large_unit(current, max_chars))
                current = ""
        else:
            chunks.extend(split_large_unit(unit, max_chars))

    if current:
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]


def make_chunk(doc, section, section_chunk_index, chunk_index, body_part):
    document_id = clean_value(doc.get("document_id"))
    header = build_header(doc, section)
    text = f"{header}\n\n{body_part}".strip()
    metadata = doc.get("metadata") or {}
    chunk_metadata = {key: metadata.get(key) for key in METADATA_KEEP if key in metadata}
    return {
        "chunk_id": f"{document_id}__{section}__{section_chunk_index:03d}",
        "document_id": document_id,
        "chunk_index": chunk_index,
        "section": section,
        "section_chunk_index": section_chunk_index,
        "text": text,
        "body_text": body_part,
        "text_char_count": len(text),
        "body_char_count": len(body_part),
        "source": doc.get("source"),
        "title": doc.get("title"),
        "url": doc.get("url"),
        "metadata": chunk_metadata,
    }


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def percentile(values, ratio):
    if not values:
        return None
    index = min(len(values) - 1, int(len(values) * ratio))
    return sorted(values)[index]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def flatten_metadata_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        return ", ".join(clean_value(item) for item in value if clean_value(item))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return clean_value(value)


def build_chroma_ready_chunk(chunk):
    metadata = {
        key: flatten_metadata_value(value)
        for key, value in (chunk.get("metadata") or {}).items()
    }
    metadata.update(
        {
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "title": flatten_metadata_value(chunk.get("title")),
            "url": flatten_metadata_value(chunk.get("url")),
            "source": flatten_metadata_value(chunk.get("source")),
            "section": chunk["section"],
            "chunk_index": chunk["chunk_index"],
            "section_chunk_index": chunk["section_chunk_index"],
            "text_char_count": chunk["text_char_count"],
            "body_char_count": chunk["body_char_count"],
        }
    )
    return {
        "id": chunk["chunk_id"],
        "text": chunk["text"],
        "metadata": metadata,
    }


def validate_chroma_metadata(rows):
    allowed_types = (str, int, float, bool)
    invalid = []
    for row in rows:
        for key, value in row.get("metadata", {}).items():
            if not isinstance(value, allowed_types):
                invalid.append(
                    {
                        "id": row.get("id"),
                        "key": key,
                        "type": type(value).__name__,
                    }
                )
                if len(invalid) >= 10:
                    return invalid
    return invalid


def write_quality_report(path, doc_reports):
    fieldnames = [
        "document_id",
        "title",
        "retrieval_text_char_count",
        "chunk_count",
        "sections",
        "max_body_char_count",
        "max_text_char_count",
        "blank_sections",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in doc_reports:
            writer.writerow(row)


def validate_chunks(chunks, expected_doc_ids):
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    duplicate_chunk_ids = len(chunk_ids) - len(set(chunk_ids))
    blank_chunks = sum(1 for chunk in chunks if not clean_value(chunk.get("text")))
    chunk_doc_ids = {chunk["document_id"] for chunk in chunks}
    missing_doc_ids = sorted(expected_doc_ids - chunk_doc_ids)
    return {
        "duplicate_chunk_id_count": duplicate_chunk_ids,
        "blank_chunk_count": blank_chunks,
        "missing_document_ids": missing_doc_ids,
    }


def build_chunks(docs, max_chars, overlap_chars):
    chunks = []
    doc_reports = []

    for doc in docs:
        document_id = clean_value(doc.get("document_id"))
        chunk_index = 0
        doc_sections = []
        blank_sections = []
        doc_chunks = []

        for section, fields in SECTION_DEFS:
            body = build_section_body(doc, fields)
            if not body:
                blank_sections.append(section)
                continue
            body_parts = split_body(body, max_chars=max_chars, overlap_chars=overlap_chars)
            for section_chunk_index, body_part in enumerate(body_parts):
                chunk = make_chunk(
                    doc=doc,
                    section=section,
                    section_chunk_index=section_chunk_index,
                    chunk_index=chunk_index,
                    body_part=body_part,
                )
                chunks.append(chunk)
                doc_chunks.append(chunk)
                doc_sections.append(section)
                chunk_index += 1

        doc_reports.append(
            {
                "document_id": document_id,
                "title": clean_value(doc.get("title")),
                "retrieval_text_char_count": len(clean_value(doc.get("retrieval_text"))),
                "chunk_count": len(doc_chunks),
                "sections": "|".join(sorted(set(doc_sections))),
                "max_body_char_count": max((chunk["body_char_count"] for chunk in doc_chunks), default=0),
                "max_text_char_count": max((chunk["text_char_count"] for chunk in doc_chunks), default=0),
                "blank_sections": "|".join(blank_sections),
            }
        )

    return chunks, doc_reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="01.document/documents.jsonl")
    parser.add_argument("--output-dir", default="05.chunks/c2_section_800")
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--overlap-chars", type=int, default=120)
    parser.add_argument("--experiment-id", default="c2_section_800")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs = read_jsonl(input_path)
    chunks, doc_reports = build_chunks(docs, args.max_chars, args.overlap_chars)

    chunks_path = output_dir / f"{args.experiment_id}_chunks.jsonl"
    chroma_ready_path = output_dir / f"{args.experiment_id}_chunks_chroma_ready.jsonl"
    manifest_path = output_dir / f"{args.experiment_id}_manifest.json"
    quality_path = output_dir / f"{args.experiment_id}_quality_report.csv"

    write_jsonl(chunks_path, chunks)
    chroma_ready_chunks = [build_chroma_ready_chunk(chunk) for chunk in chunks]
    write_jsonl(chroma_ready_path, chroma_ready_chunks)
    write_quality_report(quality_path, doc_reports)

    body_lengths = [chunk["body_char_count"] for chunk in chunks]
    text_lengths = [chunk["text_char_count"] for chunk in chunks]
    section_counts = Counter(chunk["section"] for chunk in chunks)
    doc_chunk_counts = [row["chunk_count"] for row in doc_reports]
    expected_doc_ids = {clean_value(doc.get("document_id")) for doc in docs}
    validation = validate_chunks(chunks, expected_doc_ids)
    chroma_metadata_invalid = validate_chroma_metadata(chroma_ready_chunks)

    manifest = {
        "dataset_name": "Ontong Youth MVP 400 C2 Section Chunks",
        "dataset_version": f"{args.experiment_id}_v1",
        "experiment_id": args.experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "chunking_strategy": {
            "name": "C2 section-based chunking",
            "target_field": "retrieval_text",
            "section_source": "raw_record plus metadata",
            "max_body_chars": args.max_chars,
            "overlap_chars": args.overlap_chars,
            "header_included": True,
            "long_section_split_only": True,
        },
        "counts": {
            "document_count": len(docs),
            "chunk_count": len(chunks),
            "avg_chunks_per_document": round(len(chunks) / len(docs), 3) if docs else 0,
            "min_chunks_per_document": min(doc_chunk_counts) if doc_chunk_counts else 0,
            "max_chunks_per_document": max(doc_chunk_counts) if doc_chunk_counts else 0,
            "section_counts": dict(sorted(section_counts.items())),
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
            "row_count": len(chroma_ready_chunks),
            "invalid_metadata_value_examples": chroma_metadata_invalid,
            "metadata_is_flat_scalar": not chroma_metadata_invalid,
        },
        "output_files": {
            "chunks_jsonl": str(chunks_path),
            "chunks_chroma_ready_jsonl": str(chroma_ready_path),
            "manifest": str(manifest_path),
            "quality_report_csv": str(quality_path),
        },
    }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

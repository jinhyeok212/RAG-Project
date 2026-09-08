from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT_CATALOG = Path(r"C:\말똥가리\data\processed\ontong_youth_catalog\ontong_youth_policy_catalog.csv")
DEFAULT_INPUT_RAW_JSONL = Path(r"C:\말똥가리\data\processed\ontong_youth_catalog\ontong_youth_policy_all_raw_records.jsonl")
DEFAULT_OUTPUT_DIR = Path(r"C:\말똥가리\data\processed\ontong_youth_normalization")

STANDARD_LCLSF = ["일자리", "주거", "교육", "복지문화", "참여권리"]
STANDARD_MCLSF = [
    "취업",
    "재직자",
    "창업",
    "주택 및 거주지",
    "기숙사",
    "전월세 및 주거급여 지원",
    "미래역량강화",
    "교육비지원",
    "온라인교육",
    "취약계층 및 금융지원",
    "건강",
    "예술인지원",
    "문화활동",
    "청년참여",
    "정책인프라구축",
    "청년국제교류",
    "권익보호",
]
STANDARD_KEYWORDS = [
    "대출",
    "보조금",
    "바우처",
    "금리혜택",
    "교육지원",
    "맞춤형상담서비스",
    "인턴",
    "벤처",
    "중소기업",
    "청년가장",
    "장기미취업청년",
    "공공임대주택",
    "신용회복",
    "육아",
    "출산",
    "해외진출",
    "주거지원",
]

LCLSF_ALIAS = {
    "일자리": "일자리",
    "주거": "주거",
    "교육": "교육",
    "교육･직업훈련": "교육",
    "교육·직업훈련": "교육",
    "복지문화": "복지문화",
    "금융･복지･문화": "복지문화",
    "금융·복지·문화": "복지문화",
    "참여권리": "참여권리",
    "참여･기반": "참여권리",
    "참여·기반": "참여권리",
}

MCLSF_ALIAS = {
    "취업": "취업",
    "재직자": "재직자",
    "창업": "창업",
    "주택 및 거주지": "주택 및 거주지",
    "기숙사": "기숙사",
    "전월세 및 주거급여 지원": "전월세 및 주거급여 지원",
    "미래역량강화": "미래역량강화",
    "교육비지원": "교육비지원",
    "온라인교육": "온라인교육",
    "온·오프라인교육": "온라인교육",
    "온·오프라인교육 ": "온라인교육",
    "취약계층 및 금융지원": "취약계층 및 금융지원",
    "건강": "건강",
    "예술인지원": "예술인지원",
    "문화활동": "문화활동",
    "문화활동 및 생활지원": "문화활동",
    "청년참여": "청년참여",
    "정책인프라구축": "정책인프라구축",
    "청년국제교류": "청년국제교류",
    "권익보호": "권익보호",
}

MCLSF_TO_LCLSF = {
    "취업": "일자리",
    "재직자": "일자리",
    "창업": "일자리",
    "주택 및 거주지": "주거",
    "기숙사": "주거",
    "전월세 및 주거급여 지원": "주거",
    "미래역량강화": "교육",
    "교육비지원": "교육",
    "온라인교육": "교육",
    "취약계층 및 금융지원": "복지문화",
    "건강": "복지문화",
    "예술인지원": "복지문화",
    "문화활동": "복지문화",
    "청년참여": "참여권리",
    "정책인프라구축": "참여권리",
    "청년국제교류": "참여권리",
    "권익보호": "참여권리",
}

CATEGORY_PRIORITY = ["주거", "일자리", "교육", "복지문화", "참여권리"]


def split_multi_value(value: str) -> list[str]:
    tokens = []
    for token in str(value or "").replace("\r", "\n").split(","):
        token = " ".join(token.split())
        if token:
            tokens.append(token)
    return tokens


def unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def normalize_tokens(value: str, alias: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    raw_tokens = split_multi_value(value)
    normalized = []
    unknown = []
    for token in raw_tokens:
        mapped = alias.get(token)
        if mapped:
            normalized.append(mapped)
        else:
            unknown.append(token)
    return raw_tokens, unique_preserve_order(normalized), unique_preserve_order(unknown)


def choose_primary_category(normalized_lclsf: list[str], normalized_mclsf: list[str]) -> str:
    scores = Counter()
    for category in normalized_lclsf:
        scores[category] += 1
    for middle in normalized_mclsf:
        category = MCLSF_TO_LCLSF.get(middle)
        if category:
            scores[category] += 2
    if not scores:
        return "UNKNOWN"
    return max(CATEGORY_PRIORITY, key=lambda category: (scores[category], -CATEGORY_PRIORITY.index(category)))


def choose_primary_mclsf(normalized_mclsf: list[str], primary_category: str) -> str:
    if not normalized_mclsf:
        return "UNKNOWN"
    for middle in normalized_mclsf:
        if MCLSF_TO_LCLSF.get(middle) == primary_category:
            return middle
    return normalized_mclsf[0]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_value_audit(rows: list[dict[str, str]], field: str) -> list[dict[str, Any]]:
    counter = Counter((row.get(field) or "").strip() for row in rows)
    return [{"field": field, "raw_value": value, "count": count} for value, count in counter.most_common()]


def make_token_audit(rows: list[dict[str, str]], field: str, alias: dict[str, str]) -> list[dict[str, Any]]:
    raw_counter = Counter()
    normalized_counter = Counter()
    unknown_counter = Counter()
    for row in rows:
        raw_tokens, normalized_tokens, unknown_tokens = normalize_tokens(row.get(field, ""), alias)
        raw_counter.update(raw_tokens or [""])
        normalized_counter.update(normalized_tokens or ["UNKNOWN"])
        unknown_counter.update(unknown_tokens)
    audit = []
    for token, count in raw_counter.most_common():
        audit.append(
            {
                "field": field,
                "token_type": "raw",
                "value": token,
                "normalized_value": alias.get(token, ""),
                "count": count,
            }
        )
    for token, count in normalized_counter.most_common():
        audit.append(
            {
                "field": field,
                "token_type": "normalized",
                "value": token,
                "normalized_value": token,
                "count": count,
            }
        )
    for token, count in unknown_counter.most_common():
        audit.append(
            {
                "field": field,
                "token_type": "unknown",
                "value": token,
                "normalized_value": "",
                "count": count,
            }
        )
    return audit


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    l_raw, l_norm, l_unknown = normalize_tokens(row.get("lclsfNm", ""), LCLSF_ALIAS)
    m_raw, m_norm, m_unknown = normalize_tokens(row.get("mclsfNm", ""), MCLSF_ALIAS)
    k_raw, k_norm, k_unknown = normalize_tokens(row.get("plcyKywdNm", ""), {keyword: keyword for keyword in STANDARD_KEYWORDS})
    primary_category = choose_primary_category(l_norm, m_norm)
    primary_mclsf = choose_primary_mclsf(m_norm, primary_category)
    normalized = dict(row)
    normalized.update(
        {
            "normalized_lclsf_list": json.dumps(l_norm, ensure_ascii=False),
            "normalized_mclsf_list": json.dumps(m_norm, ensure_ascii=False),
            "normalized_keywords": json.dumps(k_norm, ensure_ascii=False),
            "primary_category": primary_category,
            "primary_mclsf": primary_mclsf,
            "secondary_categories": json.dumps([value for value in l_norm if value != primary_category], ensure_ascii=False),
            "secondary_mclsf": json.dumps([value for value in m_norm if value != primary_mclsf], ensure_ascii=False),
            "raw_lclsf_tokens": json.dumps(l_raw, ensure_ascii=False),
            "raw_mclsf_tokens": json.dumps(m_raw, ensure_ascii=False),
            "raw_keyword_tokens": json.dumps(k_raw, ensure_ascii=False),
            "unknown_lclsf_tokens": json.dumps(l_unknown, ensure_ascii=False),
            "unknown_mclsf_tokens": json.dumps(m_unknown, ensure_ascii=False),
            "unknown_keyword_tokens": json.dumps(k_unknown, ensure_ascii=False),
            "is_multi_lclsf": len(l_norm) > 1,
            "is_multi_mclsf": len(m_norm) > 1,
            "normalization_status": "OK" if primary_category != "UNKNOWN" and not (l_unknown or m_unknown or k_unknown) else "CHECK",
        }
    )
    return normalized


def run(input_catalog: Path, input_raw_jsonl: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_rows = read_csv(input_catalog)
    raw_rows = read_jsonl(input_raw_jsonl)
    raw_by_policy_no = {str(row.get("plcyNo", "")).strip(): row for row in raw_rows if str(row.get("plcyNo", "")).strip()}

    normalized_catalog = [normalize_row(row) for row in catalog_rows]
    normalized_raw = []
    for row in raw_rows:
        plcy_no = str(row.get("plcyNo", "")).strip()
        catalog_match = next((item for item in normalized_catalog if item.get("plcyNo") == plcy_no), None)
        enriched = dict(row)
        if catalog_match:
            for field in [
                "document_id",
                "normalized_lclsf_list",
                "normalized_mclsf_list",
                "normalized_keywords",
                "primary_category",
                "primary_mclsf",
                "secondary_categories",
                "secondary_mclsf",
                "normalization_status",
            ]:
                enriched[field] = catalog_match.get(field, "")
        normalized_raw.append(enriched)

    category_value_audit = (
        make_value_audit(catalog_rows, "lclsfNm")
        + make_value_audit(catalog_rows, "mclsfNm")
        + make_value_audit(catalog_rows, "plcyKywdNm")
    )
    token_audit = (
        make_token_audit(catalog_rows, "lclsfNm", LCLSF_ALIAS)
        + make_token_audit(catalog_rows, "mclsfNm", MCLSF_ALIAS)
        + make_token_audit(catalog_rows, "plcyKywdNm", {keyword: keyword for keyword in STANDARD_KEYWORDS})
    )

    unknown_rows = []
    for row in normalized_catalog:
        if (
            row["primary_category"] == "UNKNOWN"
            or row["unknown_lclsf_tokens"] != "[]"
            or row["unknown_mclsf_tokens"] != "[]"
            or row["unknown_keyword_tokens"] != "[]"
        ):
            unknown_rows.append(
                {
                    "document_id": row.get("document_id", ""),
                    "plcyNo": row.get("plcyNo", ""),
                    "plcyNm": row.get("plcyNm", ""),
                    "lclsfNm": row.get("lclsfNm", ""),
                    "mclsfNm": row.get("mclsfNm", ""),
                    "plcyKywdNm": row.get("plcyKywdNm", ""),
                    "primary_category": row.get("primary_category", ""),
                    "primary_mclsf": row.get("primary_mclsf", ""),
                    "unknown_lclsf_tokens": row.get("unknown_lclsf_tokens", ""),
                    "unknown_mclsf_tokens": row.get("unknown_mclsf_tokens", ""),
                    "unknown_keyword_tokens": row.get("unknown_keyword_tokens", ""),
                    "normalization_status": row.get("normalization_status", ""),
                }
            )

    qids = [row.get("plcyNo", "") for row in normalized_catalog]
    doc_ids = [row.get("document_id", "") for row in normalized_catalog]
    primary_counts = Counter(row["primary_category"] for row in normalized_catalog)
    primary_mclsf_counts = Counter(row["primary_mclsf"] for row in normalized_catalog)
    multi_lclsf_count = sum(1 for row in normalized_catalog if row["is_multi_lclsf"])
    multi_mclsf_count = sum(1 for row in normalized_catalog if row["is_multi_mclsf"])
    quality = {
        "input_catalog_rows": len(catalog_rows),
        "input_raw_jsonl_rows": len(raw_rows),
        "normalized_catalog_rows": len(normalized_catalog),
        "normalized_raw_jsonl_rows": len(normalized_raw),
        "missing_policy_number_count": sum(1 for value in qids if not value),
        "duplicate_policy_number_count": len(qids) - len(set(qids)),
        "missing_document_id_count": sum(1 for value in doc_ids if not value),
        "duplicate_document_id_count": len(doc_ids) - len(set(doc_ids)),
        "primary_category_unknown_count": primary_counts["UNKNOWN"],
        "primary_mclsf_unknown_count": primary_mclsf_counts["UNKNOWN"],
        "multi_lclsf_row_count": multi_lclsf_count,
        "multi_mclsf_row_count": multi_mclsf_count,
        "unknown_or_check_row_count": len(unknown_rows),
    }
    quality_rows = [{"metric": key, "value": value} for key, value in quality.items()]
    for category in STANDARD_LCLSF + ["UNKNOWN"]:
        quality_rows.append({"metric": f"primary_category.{category}", "value": primary_counts[category]})
    for middle in STANDARD_MCLSF + ["UNKNOWN"]:
        quality_rows.append({"metric": f"primary_mclsf.{middle}", "value": primary_mclsf_counts[middle]})

    rules = {
        "standard_lclsf": STANDARD_LCLSF,
        "standard_mclsf": STANDARD_MCLSF,
        "standard_keywords": STANDARD_KEYWORDS,
        "lclsf_alias": LCLSF_ALIAS,
        "mclsf_alias": MCLSF_ALIAS,
        "mclsf_to_lclsf": MCLSF_TO_LCLSF,
        "category_priority_for_primary": CATEGORY_PRIORITY,
        "principles": [
            "원본 lclsfNm, mclsfNm, plcyKywdNm은 수정하지 않는다.",
            "쉼표 복합 분류는 중복 제거 후 list로 보존한다.",
            "MVP 샘플링용 대표 분류는 primary_category에 별도 저장한다.",
            "정규화 불가 값은 자동 삭제하지 않고 normalization_unknown_values.csv에 남긴다.",
        ],
    }
    manifest = {
        "dataset_name": "Ontong Youth Policy Normalized Catalog",
        "dataset_version": "ontong_youth_normalized_catalog_v1",
        "source_catalog": str(input_catalog),
        "source_raw_jsonl": str(input_raw_jsonl),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "quality": quality,
        "primary_category_counts": dict(primary_counts),
        "primary_mclsf_counts": dict(primary_mclsf_counts),
        "rules_file": str(output_dir / "normalization_rules.json"),
    }

    base_fields = list(catalog_rows[0].keys()) if catalog_rows else []
    added_fields = [
        "normalized_lclsf_list",
        "normalized_mclsf_list",
        "normalized_keywords",
        "primary_category",
        "primary_mclsf",
        "secondary_categories",
        "secondary_mclsf",
        "raw_lclsf_tokens",
        "raw_mclsf_tokens",
        "raw_keyword_tokens",
        "unknown_lclsf_tokens",
        "unknown_mclsf_tokens",
        "unknown_keyword_tokens",
        "is_multi_lclsf",
        "is_multi_mclsf",
        "normalization_status",
    ]
    write_csv(output_dir / "ontong_youth_policy_normalized_catalog.csv", normalized_catalog, base_fields + added_fields)
    write_jsonl(output_dir / "ontong_youth_policy_normalized_catalog.jsonl", normalized_catalog)
    write_jsonl(output_dir / "ontong_youth_policy_normalized_raw_records.jsonl", normalized_raw)
    write_csv(output_dir / "category_value_audit.csv", category_value_audit, ["field", "raw_value", "count"])
    write_csv(output_dir / "keyword_value_audit.csv", token_audit, ["field", "token_type", "value", "normalized_value", "count"])
    write_csv(
        output_dir / "normalization_unknown_values.csv",
        unknown_rows,
        [
            "document_id",
            "plcyNo",
            "plcyNm",
            "lclsfNm",
            "mclsfNm",
            "plcyKywdNm",
            "primary_category",
            "primary_mclsf",
            "unknown_lclsf_tokens",
            "unknown_mclsf_tokens",
            "unknown_keyword_tokens",
            "normalization_status",
        ],
    )
    write_csv(output_dir / "normalization_quality_report.csv", quality_rows, ["metric", "value"])
    write_json(output_dir / "normalization_rules.json", rules)
    write_json(output_dir / "normalization_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-catalog", type=Path, default=DEFAULT_INPUT_CATALOG)
    parser.add_argument("--input-raw-jsonl", type=Path, default=DEFAULT_INPUT_RAW_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = run(args.input_catalog, args.input_raw_jsonl, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

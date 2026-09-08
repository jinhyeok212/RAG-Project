from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    r"C:\말똥가리\data\processed\ontong_youth_normalization\ontong_youth_policy_normalized_raw_records.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(r"C:\말똥가리\data\processed\ontong_youth_mvp_400")

TARGET_COUNTS = {
    "일자리": 140,
    "복지문화": 80,
    "주거": 60,
    "교육": 60,
    "참여권리": 60,
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    return read_csv(path)


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


def text_len(row: dict[str, Any], field: str) -> int:
    return len((row.get(field) or "").strip())


def has_any(row: dict[str, Any], fields: list[str]) -> bool:
    return any((row.get(field) or "").strip() for field in fields)


def compute_quality_score(row: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons = []
    if text_len(row, "plcyNm") >= 8:
        score += 8
        reasons.append("title_ok")
    if text_len(row, "plcyExplnCn") >= 40:
        score += 14
        reasons.append("description_rich")
    elif text_len(row, "plcyExplnCn") > 0:
        score += 6
        reasons.append("description_present")
    if text_len(row, "plcySprtCn") >= 120:
        score += 26
        reasons.append("support_content_rich_full")
    elif text_len(row, "plcySprtCn") >= 40:
        score += 22
        reasons.append("support_content_rich")
    elif text_len(row, "plcySprtCn") > 0:
        score += 10
        reasons.append("support_content_present")
    if has_any(row, ["ptcpPrpTrgtCn", "addAplyQlfcCndCn", "sprtTrgtMinAge", "sprtTrgtMaxAge"]):
        score += 12
        reasons.append("target_condition_present")
    if text_len(row, "aplyYmd") > 0:
        score += 9
        reasons.append("application_period_present")
    if text_len(row, "plcyAplyMthdCn") > 0:
        score += 9
        reasons.append("application_method_present")
    if text_len(row, "sbmsnDcmntCn") > 0:
        score += 7
        reasons.append("submission_docs_present")
    if has_any(row, ["aplyUrlAddr", "refUrlAddr1"]):
        score += 7
        reasons.append("url_present")
    if has_any(row, ["etcMttrCn", "srngMthdCn", "bizPrdEtcCn"]):
        score += 4
        reasons.append("extra_guidance_present")
    if text_len(row, "plcyKywdNm") > 0:
        score += 5
        reasons.append("keyword_present")
    if (row.get("normalization_status") or "") == "OK":
        score += 5
        reasons.append("normalization_ok")
    if (row.get("primary_category") or "") == "UNKNOWN":
        score -= 100
        reasons.append("unknown_category_penalty")
    # Pure event/recruitment pages can still be useful, but keep them behind richer benefit policies.
    title = row.get("plcyNm", "")
    if any(word in title for word in ["축제", "공모전", "특강", "설명회"]):
        score -= 8
        reasons.append("event_like_penalty")
    return score, reasons


def allocate_by_middle(rows: list[dict[str, Any]], target: int) -> dict[str, int]:
    by_middle = defaultdict(list)
    for row in rows:
        by_middle[row.get("primary_mclsf") or "UNKNOWN"].append(row)
    middles = sorted(by_middle, key=lambda key: (-len(by_middle[key]), key))
    allocation = {middle: 0 for middle in middles}
    remaining = target
    # First pass: at least one per middle where possible.
    for middle in middles:
        if remaining <= 0:
            break
        allocation[middle] += 1
        remaining -= 1
    # Proportional pass.
    total = len(rows)
    fractional = []
    for middle in middles:
        raw_quota = target * (len(by_middle[middle]) / total)
        base = max(allocation[middle], int(raw_quota))
        base = min(base, len(by_middle[middle]))
        if base > allocation[middle]:
            delta = min(base - allocation[middle], remaining)
            allocation[middle] += delta
            remaining -= delta
        fractional.append((raw_quota - int(raw_quota), len(by_middle[middle]), middle))
    # Remainder goes to largest fractional/available groups.
    for _, _, middle in sorted(fractional, reverse=True):
        if remaining <= 0:
            break
        available = len(by_middle[middle]) - allocation[middle]
        if available <= 0:
            continue
        delta = min(available, remaining)
        allocation[middle] += delta
        remaining -= delta
    return allocation


def select_category(rows: list[dict[str, Any]], target: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_middle = defaultdict(list)
    for row in rows:
        by_middle[row.get("primary_mclsf") or "UNKNOWN"].append(row)
    for group in by_middle.values():
        group.sort(
            key=lambda row: (
                -int(row["document_quality_score"]),
                row.get("plcyNo", ""),
            )
        )
    allocation = allocate_by_middle(rows, target)
    selected: list[dict[str, Any]] = []
    selection_notes: list[dict[str, Any]] = []
    for middle, count in allocation.items():
        picks = by_middle[middle][:count]
        selected.extend(picks)
        selection_notes.append(
            {
                "primary_category": rows[0].get("primary_category", "") if rows else "",
                "primary_mclsf": middle,
                "available_count": len(by_middle[middle]),
                "target_count": count,
                "selected_count": len(picks),
            }
        )
    if len(selected) < target:
        selected_ids = {row["document_id"] for row in selected}
        leftovers = [row for row in rows if row["document_id"] not in selected_ids]
        leftovers.sort(key=lambda row: (-int(row["document_quality_score"]), row.get("plcyNo", "")))
        selected.extend(leftovers[: target - len(selected)])
    selected.sort(key=lambda row: (row.get("primary_category", ""), row.get("primary_mclsf", ""), -int(row["document_quality_score"])))
    return selected[:target], selection_notes


def run(input_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(input_path)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        score, reasons = compute_quality_score(row)
        new_row = dict(row)
        new_row["document_quality_score"] = score
        new_row["document_quality_reasons"] = json.dumps(reasons, ensure_ascii=False)
        enriched.append(new_row)

    selected: list[dict[str, Any]] = []
    selection_report_rows: list[dict[str, Any]] = []
    for category, target in TARGET_COUNTS.items():
        category_rows = [row for row in enriched if row.get("primary_category") == category]
        category_selected, notes = select_category(category_rows, target)
        selected.extend(category_selected)
        selection_report_rows.extend(notes)

    selected_ids = [row.get("document_id", "") for row in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("Duplicate document_id selected")

    selected_set = set(selected_ids)
    candidate_rows = []
    for row in enriched:
        candidate = dict(row)
        candidate["mvp_selected"] = "Y" if row.get("document_id") in selected_set else "N"
        candidate_rows.append(candidate)

    category_counts = Counter(row.get("primary_category", "") for row in selected)
    middle_counts = Counter(row.get("primary_mclsf", "") for row in selected)
    quality_scores = [int(row["document_quality_score"]) for row in selected]
    quality_summary = {
        "min": min(quality_scores),
        "avg": round(sum(quality_scores) / len(quality_scores), 3),
        "max": max(quality_scores),
    }
    manifest = {
        "dataset_name": "Ontong Youth MVP 400 Corpus Selection",
        "dataset_version": "ontong_youth_mvp_400_selection_v1",
        "source_normalized_catalog": str(input_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "category_target + primary_mclsf_balance + document_quality_score_desc",
        "target_counts": TARGET_COUNTS,
        "selected_count": len(selected),
        "category_counts": dict(category_counts),
        "middle_counts": dict(middle_counts),
        "quality_score_summary": quality_summary,
        "excluded_unknown_count": sum(1 for row in enriched if row.get("primary_category") == "UNKNOWN"),
        "duplicate_document_id_count": len(selected_ids) - len(set(selected_ids)),
        "output_files": {
            "mvp_catalog_csv": str(output_dir / "ontong_youth_mvp_400_catalog.csv"),
            "mvp_catalog_jsonl": str(output_dir / "ontong_youth_mvp_400_catalog.jsonl"),
            "candidate_scored_catalog_csv": str(output_dir / "ontong_youth_mvp_400_candidate_scored_catalog.csv"),
            "selection_report_csv": str(output_dir / "ontong_youth_mvp_400_selection_report.csv"),
            "manifest": str(output_dir / "ontong_youth_mvp_400_manifest.json"),
        },
    }

    base_fields = list(rows[0].keys()) if rows else []
    output_fields = base_fields + ["document_quality_score", "document_quality_reasons"]
    scored_fields = output_fields + ["mvp_selected"]
    write_csv(output_dir / "ontong_youth_mvp_400_catalog.csv", selected, output_fields)
    write_jsonl(output_dir / "ontong_youth_mvp_400_catalog.jsonl", selected)
    write_csv(output_dir / "ontong_youth_mvp_400_candidate_scored_catalog.csv", candidate_rows, scored_fields)
    write_csv(
        output_dir / "ontong_youth_mvp_400_selection_report.csv",
        selection_report_rows,
        ["primary_category", "primary_mclsf", "available_count", "target_count", "selected_count"],
    )
    write_json(output_dir / "ontong_youth_mvp_400_manifest.json", manifest)
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

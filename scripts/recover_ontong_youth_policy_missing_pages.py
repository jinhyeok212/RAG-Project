from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

import fetch_ontong_youth_policy_full as full_fetch


DEFAULT_URL = "https://www.youthcenter.go.kr/go/ythip/getPlcy"
DEFAULT_RAW_DIR = Path(r"C:\말똥가리\data\raw\ontong_youth\policy_api_full_recovered_pagesize1")
DEFAULT_PROCESSED_DIR = Path(r"C:\말똥가리\data\processed\ontong_youth_catalog")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fetch_with_retries(
    session: requests.Session,
    url: str,
    api_key: str,
    page_num: int,
    page_size: int,
    retries: int,
    sleep_seconds: float,
) -> tuple[dict[str, Any] | None, str]:
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            payload = full_fetch.fetch_page(session, url, api_key, page_num, page_size)
            return payload, ""
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            time.sleep(sleep_seconds * attempt)
    return None, last_error


def run(
    url: str,
    raw_dir: Path,
    processed_dir: Path,
    failed_pages: list[int],
    original_page_size: int,
    api_key_env: str,
    retries: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(f"Missing environment variable: {api_key_env}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    existing_records_path = processed_dir / "ontong_youth_policy_all_raw_records.jsonl"
    records = read_jsonl(existing_records_path)
    by_policy_no: dict[str, dict[str, Any]] = {
        str(row.get("plcyNo", "")).strip(): row for row in records if str(row.get("plcyNo", "")).strip()
    }

    recovered_records: list[dict[str, Any]] = []
    still_failed: list[dict[str, Any]] = []
    session = requests.Session()

    for failed_page in failed_pages:
        start_one_based = ((failed_page - 1) * original_page_size) + 1
        end_one_based = failed_page * original_page_size
        for one_page_num in range(start_one_based, end_one_based + 1):
            payload, error = fetch_with_retries(session, url, api_key, one_page_num, 1, retries, sleep_seconds)
            if payload is None:
                still_failed.append(
                    {
                        "original_failed_page": failed_page,
                        "page_num_at_page_size_1": one_page_num,
                        "error": error,
                    }
                )
                continue
            full_fetch.write_json(raw_dir / f"raw_page_size1_{one_page_num:05d}.json", payload)
            _, page_records = full_fetch.extract_result(payload)
            if not page_records:
                still_failed.append(
                    {
                        "original_failed_page": failed_page,
                        "page_num_at_page_size_1": one_page_num,
                        "error": "empty youthPolicyList",
                    }
                )
                continue
            record = dict(page_records[0])
            record["_source_page"] = one_page_num
            record["_raw_record_index"] = 1
            policy_no = str(record.get("plcyNo", "")).strip()
            if policy_no and policy_no not in by_policy_no:
                by_policy_no[policy_no] = record
                recovered_records.append(record)
            time.sleep(sleep_seconds)

    merged_records = list(by_policy_no.values())
    merged_records.sort(key=lambda row: str(row.get("plcyNo", "")), reverse=True)
    catalog_rows = [
        full_fetch.make_catalog_row(row, int(row.get("_source_page") or 0), int(row.get("_raw_record_index") or 0))
        for row in merged_records
    ]

    full_fetch.write_jsonl(processed_dir / "ontong_youth_policy_all_raw_records.jsonl", merged_records)
    full_fetch.write_csv(processed_dir / "ontong_youth_policy_catalog.csv", catalog_rows, full_fetch.CATALOG_FIELDS)

    policy_numbers = [str(row.get("plcyNo", "")).strip() for row in merged_records]
    nonblank = [number for number in policy_numbers if number]
    duplicate_numbers = sorted({number for number in nonblank if nonblank.count(number) > 1})

    manifest = {
        "source": "온통청년 청년정책 API",
        "url": url,
        "recovery_started_at": datetime.now(timezone.utc).isoformat(),
        "original_failed_pages": failed_pages,
        "original_page_size": original_page_size,
        "recovery_page_size": 1,
        "recovered_record_count": len(recovered_records),
        "still_failed_count": len(still_failed),
        "still_failed": still_failed,
        "merged_record_count": len(merged_records),
        "missing_policy_number_count": len(policy_numbers) - len(nonblank),
        "duplicate_policy_number_count": len(duplicate_numbers),
        "duplicate_policy_numbers": duplicate_numbers[:100],
        "category_counts": dict(Counter(row.get("lclsfNm", "") for row in merged_records)),
        "output_files": {
            "raw_recovered_dir": str(raw_dir),
            "raw_records_jsonl": str(processed_dir / "ontong_youth_policy_all_raw_records.jsonl"),
            "catalog_csv": str(processed_dir / "ontong_youth_policy_catalog.csv"),
            "recovery_manifest": str(processed_dir / "ontong_youth_policy_recovery_manifest.json"),
        },
    }
    full_fetch.write_json(processed_dir / "ontong_youth_policy_recovery_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--failed-pages", default="165,198,213,218,256")
    parser.add_argument("--original-page-size", type=int, default=10)
    parser.add_argument("--api-key-env", default="YOUTH_POLICY_API_KEY")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    args = parser.parse_args()
    failed_pages = [int(item.strip()) for item in args.failed_pages.split(",") if item.strip()]
    manifest = run(
        args.url,
        args.raw_dir,
        args.processed_dir,
        failed_pages,
        args.original_page_size,
        args.api_key_env,
        args.retries,
        args.sleep_seconds,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


DEFAULT_URL = "https://www.youthcenter.go.kr/go/ythip/getPlcy"
DEFAULT_RAW_DIR = Path(r"C:\말똥가리\data\raw\ontong_youth\sample")
DEFAULT_PROCESSED_DIR = Path(r"C:\말똥가리\data\processed\ontong_youth_sample")


def find_records(payload: Any) -> list[dict[str, Any]]:
    """Find the first list of dict records in a nested API response."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    preferred_paths = [
        ("result", "youthPolicyList"),
        ("result", "plcyList"),
        ("result", "list"),
        ("result", "items"),
        ("data", "youthPolicyList"),
        ("data", "plcyList"),
        ("data", "list"),
        ("data", "items"),
        ("youthPolicyList",),
        ("plcyList",),
        ("list",),
        ("items",),
    ]
    for path in preferred_paths:
        current: Any = payload
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if isinstance(current, list):
            records = [item for item in current if isinstance(item, dict)]
            if records:
                return records

    queue = [payload]
    while queue:
        current = queue.pop(0)
        if isinstance(current, dict):
            for value in current.values():
                if isinstance(value, list) and any(isinstance(item, dict) for item in value):
                    return [item for item in value if isinstance(item, dict)]
                if isinstance(value, dict):
                    queue.append(value)
        elif isinstance(current, list):
            queue.extend(item for item in current if isinstance(item, dict))
    return []


def flatten_for_csv(record: dict[str, Any]) -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in record.items():
        if isinstance(value, (dict, list)):
            flat[key] = json.dumps(value, ensure_ascii=False)
        elif value is None:
            flat[key] = ""
        else:
            flat[key] = str(value)
    return flat


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened = [flatten_for_csv(row) for row in rows]
    fieldnames = sorted({key for row in flattened for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def collect_schema(records: list[dict[str, Any]]) -> dict[str, Any]:
    key_counts: dict[str, int] = {}
    type_examples: dict[str, str] = {}
    preview_examples: dict[str, str] = {}
    for record in records:
        for key, value in record.items():
            key_counts[key] = key_counts.get(key, 0) + 1
            type_examples.setdefault(key, type(value).__name__)
            if key not in preview_examples and value not in (None, ""):
                text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
                preview_examples[key] = text[:200]
    return {
        "record_count": len(records),
        "keys": sorted(key_counts),
        "key_counts": key_counts,
        "type_examples": type_examples,
        "preview_examples": preview_examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a small Ontong Youth policy API sample.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--page-num", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=10)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--api-key-env", default="YOUTH_POLICY_API_KEY")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing environment variable: {args.api_key_env}")

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "apiKeyNm": api_key,
        "pageNum": args.page_num,
        "pageSize": args.page_size,
        "rtnType": "json",
    }
    response = requests.get(args.url, params=params, timeout=30)
    response.raise_for_status()

    try:
        payload: Any = response.json()
    except ValueError:
        payload = {"raw_text": response.text}

    records = find_records(payload)
    schema = collect_schema(records)
    manifest = {
        "source": "온통청년 청년정책 API",
        "url": args.url,
        "params": {
            "pageNum": args.page_num,
            "pageSize": args.page_size,
            "rtnType": "json",
            "apiKeyNm": "<redacted>",
        },
        "status_code": response.status_code,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "raw_response_file": str(args.raw_dir / f"youth_policy_page_{args.page_num:04d}_raw.json"),
        "sample_jsonl_file": str(args.processed_dir / "youth_policy_sample.jsonl"),
        "sample_csv_file": str(args.processed_dir / "youth_policy_sample.csv"),
        "schema_file": str(args.processed_dir / "youth_policy_sample_schema.json"),
        "record_count": len(records),
        "top_level_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
        "schema": schema,
    }

    write_json(args.raw_dir / f"youth_policy_page_{args.page_num:04d}_raw.json", payload)
    write_jsonl(args.processed_dir / "youth_policy_sample.jsonl", records)
    write_csv(args.processed_dir / "youth_policy_sample.csv", records)
    write_json(args.processed_dir / "youth_policy_sample_schema.json", schema)
    write_json(args.processed_dir / "youth_policy_sample_manifest.json", manifest)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

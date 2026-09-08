from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


DEFAULT_URL = "https://www.youthcenter.go.kr/go/ythip/getPlcy"
DEFAULT_RAW_DIR = Path(r"C:\말똥가리\data\raw\ontong_youth\policy_api_full")
DEFAULT_PROCESSED_DIR = Path(r"C:\말똥가리\data\processed\ontong_youth_catalog")


CATALOG_FIELDS = [
    "document_id",
    "plcyNo",
    "plcyNm",
    "lclsfNm",
    "mclsfNm",
    "plcyKywdNm",
    "plcyExplnCn",
    "plcySprtCn_preview",
    "aplyYmd",
    "bizPrdBgngYmd",
    "bizPrdEndYmd",
    "sprvsnInstCdNm",
    "operInstCdNm",
    "aplyUrlAddr",
    "refUrlAddr1",
    "sprtTrgtMinAge",
    "sprtTrgtMaxAge",
    "earnMinAmt",
    "earnMaxAmt",
    "source_page",
    "raw_record_index",
]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def extract_result(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = payload.get("result")
    if not isinstance(result, dict):
        return {}, []
    paging = result.get("pagging") or result.get("paging") or {}
    records = result.get("youthPolicyList") or []
    if not isinstance(paging, dict):
        paging = {}
    if not isinstance(records, list):
        records = []
    return paging, [record for record in records if isinstance(record, dict)]


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_space(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", "\n").split())


def make_document_id(plcy_no: str) -> str:
    return f"ontong_youth_{plcy_no}"


def make_catalog_row(record: dict[str, Any], page_num: int, index: int) -> dict[str, Any]:
    plcy_no = str(record.get("plcyNo", "")).strip()
    return {
        "document_id": make_document_id(plcy_no) if plcy_no else "",
        "plcyNo": plcy_no,
        "plcyNm": record.get("plcyNm", ""),
        "lclsfNm": record.get("lclsfNm", ""),
        "mclsfNm": record.get("mclsfNm", ""),
        "plcyKywdNm": record.get("plcyKywdNm", ""),
        "plcyExplnCn": normalize_space(record.get("plcyExplnCn", "")),
        "plcySprtCn_preview": normalize_space(record.get("plcySprtCn", ""))[:500],
        "aplyYmd": record.get("aplyYmd", ""),
        "bizPrdBgngYmd": record.get("bizPrdBgngYmd", ""),
        "bizPrdEndYmd": record.get("bizPrdEndYmd", ""),
        "sprvsnInstCdNm": record.get("sprvsnInstCdNm", ""),
        "operInstCdNm": record.get("operInstCdNm", ""),
        "aplyUrlAddr": record.get("aplyUrlAddr", ""),
        "refUrlAddr1": record.get("refUrlAddr1", ""),
        "sprtTrgtMinAge": record.get("sprtTrgtMinAge", ""),
        "sprtTrgtMaxAge": record.get("sprtTrgtMaxAge", ""),
        "earnMinAmt": record.get("earnMinAmt", ""),
        "earnMaxAmt": record.get("earnMaxAmt", ""),
        "source_page": page_num,
        "raw_record_index": index,
    }


def fetch_page(session: requests.Session, url: str, api_key: str, page_num: int, page_size: int) -> dict[str, Any]:
    params = {
        "apiKeyNm": api_key,
        "pageNum": page_num,
        "pageSize": page_size,
        "rtnType": "json",
    }
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def run(url: str, raw_dir: Path, processed_dir: Path, page_size: int, sleep_seconds: float, api_key_env: str) -> dict[str, Any]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(f"Missing environment variable: {api_key_env}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    started_at = datetime.now(timezone.utc).isoformat()
    first_payload = fetch_page(session, url, api_key, 1, page_size)
    write_json(raw_dir / "raw_page_0001.json", first_payload)
    first_paging, first_records = extract_result(first_payload)
    total_count = as_int(first_paging.get("totCount"), len(first_records))
    total_pages = max(1, math.ceil(total_count / page_size))

    all_records: list[dict[str, Any]] = []
    catalog_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(first_records, start=1):
        enriched = dict(record)
        enriched["_source_page"] = 1
        enriched["_raw_record_index"] = idx
        all_records.append(enriched)
        catalog_rows.append(make_catalog_row(record, 1, idx))

    failed_pages: list[dict[str, Any]] = []
    for page_num in range(2, total_pages + 1):
        try:
            payload = fetch_page(session, url, api_key, page_num, page_size)
            write_json(raw_dir / f"raw_page_{page_num:04d}.json", payload)
            _, records = extract_result(payload)
            for idx, record in enumerate(records, start=1):
                enriched = dict(record)
                enriched["_source_page"] = page_num
                enriched["_raw_record_index"] = idx
                all_records.append(enriched)
                catalog_rows.append(make_catalog_row(record, page_num, idx))
        except Exception as exc:  # noqa: BLE001
            failed_pages.append({"page": page_num, "error": repr(exc)})
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    qnos = [str(row.get("plcyNo", "")).strip() for row in all_records]
    nonblank_qnos = [qno for qno in qnos if qno]
    duplicate_policy_numbers = sorted({qno for qno in nonblank_qnos if nonblank_qnos.count(qno) > 1})

    write_jsonl(processed_dir / "ontong_youth_policy_all_raw_records.jsonl", all_records)
    write_csv(processed_dir / "ontong_youth_policy_catalog.csv", catalog_rows, CATALOG_FIELDS)

    manifest = {
        "source": "온통청년 청년정책 API",
        "url": url,
        "params": {
            "pageSize": page_size,
            "rtnType": "json",
            "apiKeyNm": "<redacted>",
        },
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "total_count_from_api": total_count,
        "total_pages_expected": total_pages,
        "records_collected": len(all_records),
        "catalog_rows": len(catalog_rows),
        "failed_pages": failed_pages,
        "missing_policy_number_count": len(qnos) - len(nonblank_qnos),
        "duplicate_policy_number_count": len(duplicate_policy_numbers),
        "duplicate_policy_numbers": duplicate_policy_numbers[:100],
        "output_files": {
            "raw_records_jsonl": str(processed_dir / "ontong_youth_policy_all_raw_records.jsonl"),
            "catalog_csv": str(processed_dir / "ontong_youth_policy_catalog.csv"),
            "manifest": str(processed_dir / "ontong_youth_policy_collection_manifest.json"),
        },
    }
    write_json(processed_dir / "ontong_youth_policy_collection_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch all Ontong Youth policy API records and create a catalog.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--api-key-env", default="YOUTH_POLICY_API_KEY")
    args = parser.parse_args()
    manifest = run(args.url, args.raw_dir, args.processed_dir, args.page_size, args.sleep_seconds, args.api_key_env)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from src.common import compact_text, ensure_dirs, load_config, write_json
from src.data.privacy_masking import detect_privacy_patterns


def csv_files(raw_root: Path, split: str) -> list[Path]:
    return sorted((raw_root / split).glob("*.csv"))


def profile_csv(path: Path, chunksize: int, max_rows: int | None = None) -> dict:
    total = 0
    nulls: Counter[str] = Counter()
    uniques: dict[str, set] = defaultdict(set)
    sample_values: dict[str, list[str]] = defaultdict(list)
    privacy_hits: Counter[str] = Counter()
    too_long = 0
    duplicate_texts: Counter[str] = Counter()
    columns: list[str] = []
    dtypes: dict[str, str] = {}

    for chunk in pd.read_csv(path, chunksize=chunksize, dtype=str, keep_default_na=False):
        if max_rows and total >= max_rows:
            break
        if max_rows and total + len(chunk) > max_rows:
            chunk = chunk.head(max_rows - total)
        if not columns:
            columns = list(chunk.columns)
            dtypes = {col: "string" for col in columns}
        total += len(chunk)
        for col in columns:
            series = chunk[col].map(compact_text)
            nulls[col] += int((series == "").sum())
            for value in series[series != ""].head(20):
                if len(uniques[col]) < 5000:
                    uniques[col].add(value)
                if len(sample_values[col]) < 8 and value not in sample_values[col]:
                    sample_values[col].append(value)
        text_col = "발화문" if "발화문" in chunk.columns else columns[0]
        for text in chunk[text_col].map(compact_text).head(10000):
            if len(text) > 300:
                too_long += 1
            duplicate_texts[text] += 1
            for label in detect_privacy_patterns(text):
                privacy_hits[label] += 1

    return {
        "path": str(path),
        "rows": total,
        "columns": columns,
        "dtypes": dtypes,
        "missing_ratio": {col: (nulls[col] / total if total else 0) for col in columns},
        "unique_counts_limited": {col: len(uniques[col]) for col in columns},
        "sample_values": sample_values,
        "too_long_utterance_count": too_long,
        "exact_duplicate_utterance_count": sum(c - 1 for t, c in duplicate_texts.items() if t and c > 1),
        "privacy_pattern_hits": dict(privacy_hits),
        "profiled_rows": total,
        "profile_limited": bool(max_rows and total >= max_rows),
    }


def inspect_archives(raw_root: Path) -> dict:
    archive_info = []
    for path in sorted(raw_root.glob("*.zip")):
        with zipfile.ZipFile(path) as zf:
            archive_info.append({"zip": str(path), "files": zf.namelist()})
    return {"zip_files": archive_info}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    raw_root = Path(cfg["paths"]["raw_root"])
    report_dir = Path(cfg["paths"]["report_dir"])
    ensure_dirs(report_dir)

    chunksize = int(cfg["data"].get("chunksize", 50000))
    max_rows = cfg["data"].get("profile_max_rows_per_file")
    max_rows = int(max_rows) if max_rows else None
    profiles = {"archives": inspect_archives(raw_root), "splits": {}, "schema_differences": {}}
    for split in ["Training", "Validation"]:
        split_profiles = []
        for path in csv_files(raw_root, split):
            split_profiles.append(profile_csv(path, chunksize, max_rows=max_rows))
        profiles["splits"][split] = split_profiles

    train_cols = {Path(p["path"]).name: p["columns"] for p in profiles["splits"].get("Training", [])}
    val_cols = {Path(p["path"]).name.replace("_validation", "_train"): p["columns"] for p in profiles["splits"].get("Validation", [])}
    for name, cols in train_cols.items():
        profiles["schema_differences"][name] = {
            "train_only": sorted(set(cols) - set(val_cols.get(name, []))),
            "validation_only": sorted(set(val_cols.get(name, [])) - set(cols)),
        }

    write_json(report_dir / "data_profile.json", profiles)
    md = ["# 데이터 구조 분석", ""]
    md.append("## 파일 목록")
    for split, rows in profiles["splits"].items():
        md.append(f"### {split}")
        for p in rows:
            md.append(f"- `{Path(p['path']).name}`: {p['rows']:,} rows, {len(p['columns'])} columns")
    md.append("")
    md.append("## 확인된 공통 컬럼")
    all_cols = profiles["splits"]["Training"][0]["columns"] if profiles["splits"].get("Training") else []
    md.append(", ".join(f"`{c}`" for c in all_cols))
    md.append("")
    md.append("## 상담/QA 복원에 사용 가능한 컬럼")
    for col in ["상담번호", "상담내순번", "QA번호", "QA여부", "발화자", "발화문", "카테고리", "인텐트"]:
        md.append(f"- `{col}`")
    md.append("")
    md.append("## 개인정보 추정 패턴")
    totals = Counter()
    for split_rows in profiles["splits"].values():
        for p in split_rows:
            totals.update(p.get("privacy_pattern_hits", {}))
    if totals:
        for k, v in totals.items():
            md.append(f"- {k}: {v:,}")
    else:
        md.append("- 샘플링/청크 분석 범위에서 정규식 기반 개인정보 패턴은 확인되지 않음")
    md.append("")
    md.append("## Train/Validation 스키마 차이")
    diffs = profiles["schema_differences"]
    if all(not d["train_only"] and not d["validation_only"] for d in diffs.values()):
        md.append("- 모든 대응 파일에서 컬럼 스키마가 동일함")
    else:
        for name, diff in diffs.items():
            if diff["train_only"] or diff["validation_only"]:
                md.append(f"- `{name}`: train_only={diff['train_only']}, validation_only={diff['validation_only']}")
    (report_dir / "data_profile.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()

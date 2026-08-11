"""검색 실험에서 공통으로 사용하는 입출력 및 검증 함수."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypeVar

import pandas as pd
from dotenv import load_dotenv

T = TypeVar("T")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_environment() -> None:
    """프로젝트 루트의 .env를 읽습니다."""
    load_dotenv(PROJECT_ROOT / ".env")


def project_path(value: str) -> Path:
    """환경변수의 상대경로를 프로젝트 루트 기준 절대경로로 바꿉니다."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def env_int(name: str, default: int) -> int:
    """양의 정수 환경변수를 안전하게 읽습니다."""
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}은 정수여야 합니다: {raw}") from exc
    if value <= 0:
        raise ValueError(f"{name}은 1 이상이어야 합니다: {value}")
    return value


def clean_text(value: object) -> str:
    """결측값을 빈 문자열로 바꾸고 앞뒤 공백을 제거합니다."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def read_and_validate_csv(path: Path, required_columns: Sequence[str]) -> pd.DataFrame:
    """UTF-8 CSV를 읽고 필수 컬럼 및 결측값을 검사합니다."""
    if not path.exists():
        raise FileNotFoundError(f"CSV 파일이 없습니다: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing_columns = [name for name in required_columns if name not in frame.columns]
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
    for column in required_columns:
        frame[column] = frame[column].map(clean_text)
    empty_counts = {column: int(frame[column].eq("").sum()) for column in required_columns}
    invalid = {key: value for key, value in empty_counts.items() if value > 0}
    if invalid:
        raise ValueError(f"필수 컬럼에 빈 값이 있습니다: {invalid}")
    return frame


def batches(values: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    """목록을 지정한 크기의 작은 묶음으로 나눕니다."""
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def ensure_unique(values: Iterable[str], label: str) -> None:
    """식별자 중복을 검사합니다."""
    items = list(values)
    duplicate_count = len(items) - len(set(items))
    if duplicate_count:
        raise ValueError(f"{label}에 중복이 {duplicate_count}개 있습니다.")

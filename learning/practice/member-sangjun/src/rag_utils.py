"""RAG 실험에서 공통으로 사용할 함수 모음."""

from __future__ import annotations

import csv
import json
import re
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
SAMPLE_DIR = DATA_DIR / "samples"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DIR = PROJECT_ROOT / "vector_store"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def ensure_project_dirs() -> None:
    """실험에서 사용할 출력 폴더를 만든다."""
    for path in [SAMPLE_DIR, PROCESSED_DIR, VECTOR_DIR / "faiss", OUTPUT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def default_normal_data_path() -> Path:
    """AIHub 기계독해 Normal zip 또는 JSON 파일의 기본 위치를 찾는다."""
    expected = DATA_DIR / "기계독해" / "기계독해분야" / "01.Normal.zip"
    if expected.exists():
        return expected

    matches = list(DATA_DIR.rglob("01.Normal.zip"))
    if matches:
        return matches[0]

    json_matches = list(DATA_DIR.rglob("ko_nia_normal_squad_all.json"))
    if json_matches:
        return json_matches[0]

    raise FileNotFoundError(
        "01.Normal.zip 또는 ko_nia_normal_squad_all.json을 찾지 못했습니다. "
        "Data/기계독해 폴더 아래에 AIHub 원본 데이터를 넣어주세요."
    )


def load_json_from_path(path: Path) -> dict[str, Any]:
    """zip 또는 JSON 파일에서 AIHub 데이터를 읽는다."""
    if path.suffix.lower() == ".zip":
        return load_json_from_zip(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_json_from_zip(zip_path: Path) -> dict[str, Any]:
    """zip 안에 들어있는 첫 JSON 파일을 읽는다."""
    with zipfile.ZipFile(zip_path) as zf:
        json_names = [name for name in zf.namelist() if name.lower().endswith(".json")]
        if not json_names:
            raise ValueError(f"zip 안에 JSON 파일이 없습니다: {zip_path}")

        with zf.open(json_names[0]) as f:
            return json.load(f)


def normalize_whitespace(text: str) -> str:
    """여러 공백과 줄바꿈을 한 칸으로 정리한다."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_match(text: str) -> str:
    """정답 포함 여부를 확인하기 위해 공백과 대소문자 차이를 줄인다."""
    return re.sub(r"\s+", "", text).lower()


def answer_in_text(answer: str, text: str) -> bool:
    """검색된 청크 안에 정답 문자열이 들어있는지 확인한다."""
    answer_norm = normalize_for_match(answer)
    text_norm = normalize_for_match(text)
    return bool(answer_norm) and answer_norm in text_norm


def chunk_text(text: str, chunk_size: int, overlap: int = 50) -> list[str]:
    """긴 문서를 일정 글자 수의 청크로 나눈다."""
    text = normalize_whitespace(text)
    if chunk_size <= 0:
        raise ValueError("chunk_size는 1 이상이어야 합니다.")
    if overlap < 0:
        raise ValueError("overlap은 0 이상이어야 합니다.")
    if overlap >= chunk_size:
        raise ValueError("overlap은 chunk_size보다 작아야 합니다.")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSON Lines 파일을 읽는다."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """dict 목록을 JSON Lines 파일로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    """CSV 파일을 dict 목록으로 읽는다."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """dict 목록을 CSV 파일로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def e5_passage(text: str) -> str:
    """E5 계열 모델이 문서 청크를 잘 이해하도록 접두어를 붙인다."""
    return f"passage: {text}"


def e5_query(text: str) -> str:
    """E5 계열 모델이 질문을 잘 이해하도록 접두어를 붙인다."""
    return f"query: {text}"


def write_faiss_index(faiss_module: Any, index: Any, path: Path) -> None:
    """Windows 한글 경로 문제를 피해서 FAISS 인덱스를 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(tempfile.gettempdir()) / f"rag_faiss_{uuid.uuid4().hex}.faiss"
    try:
        faiss_module.write_index(index, str(temp_path))
        shutil.move(str(temp_path), path)
    finally:
        temp_path.unlink(missing_ok=True)


def read_faiss_index(faiss_module: Any, path: Path) -> Any:
    """Windows 한글 경로 문제를 피해서 FAISS 인덱스를 읽는다."""
    temp_path = Path(tempfile.gettempdir()) / f"rag_faiss_{uuid.uuid4().hex}.faiss"
    try:
        shutil.copyfile(path, temp_path)
        return faiss_module.read_index(str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)

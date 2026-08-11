"""JSON 설정을 읽고 상대 경로를 프로젝트 폴더 기준 절대 경로로 바꾼다."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """코드에서 사용할 설정값 모음.

    실제 DB로 전환할 때 코드가 아니라 config.json만 수정할 수 있도록
    DB 위치, 컬렉션, 임베딩 모델, Top-k를 한곳에 모은다.
    """

    db_path: Path
    collection_name: str
    embedding_model: str
    distance_metric: str
    chunk_id_field: str
    document_id_field: str
    top_k_values: list[int]
    chunks_path: Path
    questions_path: Path
    output_dir: Path


def load_settings(config_path: str | Path) -> Settings:
    """config.json을 읽어 검증하고 Settings 객체로 반환한다."""
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    top_k_values = raw["top_k_values"]
    if not top_k_values or any(not isinstance(k, int) or k <= 0 for k in top_k_values):
        raise ValueError("top_k_values에는 1 이상의 정수만 넣어야 합니다.")

    distance_metric = raw.get("distance_metric", "cosine").lower()
    if distance_metric not in {"cosine", "l2", "ip"}:
        raise ValueError("distance_metric은 Chroma가 지원하는 cosine, l2, ip 중 하나여야 합니다.")

    metadata_fields = raw.get("metadata_fields", {})

    base_dir = config_path.parent

    def resolve_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (base_dir / path).resolve()

    return Settings(
        db_path=resolve_path(raw["db_path"]),
        collection_name=raw["collection_name"],
        embedding_model=raw["embedding_model"],
        distance_metric=distance_metric,
        chunk_id_field=metadata_fields.get("chunk_id", "chunk_id"),
        document_id_field=metadata_fields.get("document_id", "document_id"),
        top_k_values=top_k_values,
        chunks_path=resolve_path(raw["chunks_path"]),
        questions_path=resolve_path(raw["questions_path"]),
        output_dir=resolve_path(raw["output_dir"]),
    )

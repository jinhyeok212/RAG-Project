"""실제 DB가 오기 전 Retriever 검증용 샘플 Chroma DB를 임시로 만든다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import load_settings
from src.embedding import TextEmbedder
from src.vector_store import ChromaStore


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """한 줄에 JSON 객체 하나가 있는 JSONL 파일을 읽는다."""
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main(config_path: str) -> None:
    settings = load_settings(config_path)
    chunks = read_jsonl(settings.chunks_path)
    if not chunks:
        raise ValueError("저장할 청크가 없습니다.")

    # 데이터마다 달라질 수 있는 부가 정보는 metadata 아래에 두되,
    # 검색 결과에 꼭 필요한 두 ID는 항상 Chroma metadata에 함께 저장한다.
    ids = [item["chunk_id"] for item in chunks]
    texts = [item["text"] for item in chunks]
    metadatas = [
        {
            **item.get("metadata", {}),
            settings.chunk_id_field: item["chunk_id"],
            settings.document_id_field: item["document_id"],
        }
        for item in chunks
    ]

    embedder = TextEmbedder(settings.embedding_model)
    embeddings = embedder.embed(texts)
    store = ChromaStore(settings)
    store.upsert_chunks(ids, texts, embeddings, metadatas)

    print(f"완료: {len(chunks)}개 청크 저장")
    print(f"DB 경로: {settings.db_path}")
    print(f"Collection: {settings.collection_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="샘플 Chroma DB 구축")
    parser.add_argument("--config", default="config.json", help="설정 JSON 경로")
    args = parser.parse_args()
    main(args.config)

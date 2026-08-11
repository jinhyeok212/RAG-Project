"""Chroma 연결과 저장/검색만 담당한다."""

from __future__ import annotations

from typing import Any

import chromadb

from .config import Settings


class ChromaStore:
    """Retriever가 Chroma 세부 API에 강하게 결합되지 않게 하는 작은 경계."""

    def __init__(self, settings: Settings) -> None:
        settings.db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(settings.db_path))
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": settings.distance_metric},
        )

    def upsert_chunks(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """같은 chunk_id가 있으면 갱신하고, 없으면 새로 저장한다."""
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, query_embedding: list[float], top_k: int) -> dict[str, Any]:
        """질문 벡터와 가까운 청크를 top_k개 반환한다."""
        available = self.collection.count()
        if available == 0:
            raise RuntimeError(
                "컬렉션이 비어 있습니다. 실제 DB 설정을 확인하거나, "
                "학습용이라면 build_sample_db.py를 먼저 실행하세요."
            )

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, available),
            include=["documents", "metadatas", "distances"],
        )

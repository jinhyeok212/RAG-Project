"""Chroma 컬렉션 생성, 저장, 검색을 담당하는 모듈."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection

from src.retrieval_utils import batches, load_environment, project_path


class ChromaQuestionStore:
    """cosine 거리 기반 질문 벡터 저장소."""

    def __init__(self, path: str | None = None, collection_name: str | None = None) -> None:
        load_environment()
        configured_path = path or os.getenv("CHROMA_PATH", "./chroma_db")
        self.path: Path = project_path(configured_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION", "super_questions"
        ).strip()
        if not self.collection_name:
            raise ValueError("CHROMA_COLLECTION이 비어 있습니다.")
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "description": "슈퍼 Training 질문 baseline"},
        )

    def upsert(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: np.ndarray,
        metadatas: Sequence[dict[str, str]],
        batch_size: int = 500,
    ) -> None:
        """동일 ID는 갱신되도록 여러 묶음으로 upsert합니다."""
        if not (len(ids) == len(documents) == len(embeddings) == len(metadatas)):
            raise ValueError("ids, documents, embeddings, metadatas 길이가 다릅니다.")
        indices = list(range(len(ids)))
        for index_batch in batches(indices, batch_size):
            selected = list(index_batch)
            self.collection.upsert(
                ids=[ids[i] for i in selected],
                documents=[documents[i] for i in selected],
                embeddings=embeddings[selected].tolist(),
                metadatas=[metadatas[i] for i in selected],
            )

    def query(self, embedding: np.ndarray, top_k: int, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """낮은 cosine 거리 순으로 검색 결과를 평탄한 사전 목록으로 반환합니다."""
        if self.collection.count() == 0:
            return []
        count = min(top_k, self.collection.count())
        result = self.collection.query(
            query_embeddings=[embedding.astype(np.float32).tolist()],
            n_results=count,
            include=["documents", "metadatas", "distances"],
            where=where,
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            {
                "rank": index + 1,
                "document_id": ids[index],
                "question": documents[index] or "",
                "metadata": metadatas[index] or {},
                "distance": float(distances[index]),
            }
            for index in range(len(ids))
        ]

    @property
    def count(self) -> int:
        """현재 컬렉션 문서 수."""
        return int(self.collection.count())

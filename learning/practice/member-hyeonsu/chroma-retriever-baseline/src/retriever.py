"""질문 임베딩, Top-k 검색, 구간별 시간 측정을 묶는다."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from .embedding import TextEmbedder
from .vector_store import ChromaStore


class Retriever:
    """질문 하나를 받아 검색 결과 레코드 목록을 만드는 baseline Retriever."""

    def __init__(
        self,
        embedder: TextEmbedder,
        store: ChromaStore,
        distance_metric: str,
        chunk_id_field: str = "chunk_id",
        document_id_field: str = "document_id",
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.distance_metric = distance_metric.lower()
        self.chunk_id_field = chunk_id_field
        self.document_id_field = document_id_field

    def retrieve(self, question_id: str, question: str, top_k: int) -> list[dict[str, Any]]:
        """질문 임베딩과 DB 검색을 실행하고 각 단계 시간을 밀리초로 기록한다."""
        if not question.strip():
            raise ValueError("question은 빈 문자열일 수 없습니다.")
        if top_k <= 0:
            raise ValueError("top_k는 1 이상이어야 합니다.")

        total_started = perf_counter()

        embedding_started = perf_counter()
        question_vector = self.embedder.embed([question])[0]
        embedding_ms = (perf_counter() - embedding_started) * 1000

        search_started = perf_counter()
        raw = self.store.query(question_vector, top_k)
        search_ms = (perf_counter() - search_started) * 1000

        results: list[dict[str, Any]] = []

        # Chroma는 질문을 여러 개 받을 수 있어 결과가 이중 리스트다.
        # 여기서는 질문 하나만 검색하므로 첫 번째([0]) 결과만 읽는다.
        for rank, (chunk_id, text, metadata, distance) in enumerate(
            zip(
                raw["ids"][0],
                raw["documents"][0],
                raw["metadatas"][0],
                raw["distances"][0],
            ),
            start=1,
        ):
            distance = float(distance)

            # 1 - distance를 similarity처럼 해석하는 것은 cosine distance일 때만 맞다.
            # l2나 ip 등 다른 거리 방식에서는 임의 변환하지 않고 JSON null(None)을 남긴다.
            similarity = 1.0 - distance if self.distance_metric == "cosine" else None

            results.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "top_k": top_k,
                    "rank": rank,
                    "chunk_id": metadata.get(self.chunk_id_field, chunk_id),
                    "document_id": metadata.get(self.document_id_field),
                    "text": text,
                    # 원본 distance는 거리 방식과 관계없이 항상 보존한다.
                    "distance": distance,
                    "similarity": similarity,
                    "embedding_time_ms": embedding_ms,
                    "search_time_ms": search_ms,
                }
            )

        # 결과 변환까지 끝난 뒤 측정해야 '전체 처리 시간'이라는 의미에 맞는다.
        total_ms = (perf_counter() - total_started) * 1000
        for result in results:
            result["total_time_ms"] = total_ms

        return results

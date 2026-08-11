"""예측 인텐트로 기존 Chroma 검색 범위를 제한하는 재사용 가능한 검색기."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Sequence

import chromadb
import numpy as np

from src.retrieval_utils import load_environment, project_path


class IntentFilteredRetriever:
    """기존 컬렉션을 읽기만 하며 full/strict/fallback 전략을 제공합니다."""

    STRATEGIES = {"full", "top1_strict", "top1_fallback", "top3_strict", "top3_fallback"}

    def __init__(self, collection_name: str = "super_questions", chroma_path: str | None = None) -> None:
        load_environment()
        path = project_path(chroma_path or os.getenv("CHROMA_PATH", "./chroma_db"))
        self.client = chromadb.PersistentClient(path=str(path))
        names = {collection.name for collection in self.client.list_collections()}
        if collection_name not in names:
            raise RuntimeError(f"기존 Chroma 컬렉션 '{collection_name}'이 없습니다.")
        # get_collection만 사용하므로 컬렉션을 생성하거나 수정하지 않습니다.
        self.collection = self.client.get_collection(collection_name)
        self.collection_name = collection_name

    @staticmethod
    def _normalize(raw: dict[str, Any]) -> list[dict[str, Any]]:
        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        rows = []
        for index, document_id in enumerate(ids):
            metadata = metadatas[index] or {}
            rows.append({
                "document_id": document_id,
                "question": documents[index] or "",
                "answer": str(metadata.get("answer", "") or ""),
                "intent": str(metadata.get("intent", "") or "").strip(),
                "category": str(metadata.get("category", "") or ""),
                "distance": float(distances[index]),
            })
        return rows

    def _query(self, embedding: np.ndarray, count: int, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding.astype(np.float32).tolist()],
            "n_results": min(count, self.collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where
        return self._normalize(self.collection.query(**kwargs))

    @staticmethod
    def _merge(*groups: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """document_id 중복을 제거하고 원래 Chroma distance 오름차순을 유지합니다."""
        best: dict[str, dict[str, Any]] = {}
        for group in groups:
            for item in group:
                current = best.get(item["document_id"])
                if current is None or item["distance"] < current["distance"]:
                    best[item["document_id"]] = item
        return sorted(best.values(), key=lambda item: (item["distance"], item["document_id"]))

    def _intent_query(self, embedding: np.ndarray, intents: Sequence[str], count: int) -> list[dict[str, Any]]:
        clean = list(dict.fromkeys(intent.strip() for intent in intents if intent.strip()))
        if not clean:
            return []
        if len(clean) == 1:
            return self._query(embedding, count, {"intent": {"$eq": clean[0]}})
        try:
            return self._query(embedding, count, {"intent": {"$in": clean}})
        except Exception:
            # Chroma 버전이 $in을 지원하지 않으면 인텐트별 검색 후 distance로 합칩니다.
            return self._merge(*[self._query(embedding, count, {"intent": {"$eq": intent}}) for intent in clean])[:count]

    def search(
        self, embedding: np.ndarray, strategy: str, predicted_intents: Sequence[str],
        top_k: int = 5, evaluation_depth: int = 20,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
        """표시용 Top-K와 MRR 계산용 후보, 실제 검색 시간을 반환합니다."""
        if strategy not in self.STRATEGIES:
            raise ValueError(f"지원하지 않는 검색 전략입니다: {strategy}")
        started = time.perf_counter()
        depth = max(top_k, evaluation_depth)
        if strategy == "full":
            candidates = self._query(embedding, depth)
        else:
            intent_count = 1 if strategy.startswith("top1") else 3
            restricted = self._intent_query(embedding, predicted_intents[:intent_count], depth)
            candidates = restricted
            if strategy.endswith("fallback") and len(restricted[:top_k]) < top_k:
                candidates = self._merge(restricted, self._query(embedding, depth))[:depth]
        elapsed_ms = (time.perf_counter() - started) * 1000
        candidates = self._merge(candidates)[:depth]
        return candidates[:top_k], candidates, elapsed_ms

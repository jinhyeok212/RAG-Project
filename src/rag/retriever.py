"""Read-only Top-k retrieval and result serialization.

This module deliberately does not create, rebuild, or mutate a Chroma
collection. An embedder and an already-loaded collection are injected by the
caller so retrieval can be tested without loading the real model or database.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


class QueryEmbedder(Protocol):
    def embed_query(self, query: str) -> Any: ...


class ReadOnlyCollection(Protocol):
    def count(self) -> int: ...

    def query(self, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class RetrievalConfig:
    embedding_model: str = "nlpai-lab/KURE-v1"
    index_version: str = "index_baseline_kure_v1"
    distance_metric: str = "cosine"
    evaluation_max_k: int = 5

    def __post_init__(self) -> None:
        if self.evaluation_max_k < 1:
            raise ValueError("evaluation_max_k must be at least 1")
        if self.distance_metric != "cosine":
            raise ValueError("The baseline collection requires cosine distance")


class KUREQueryEmbedder:
    """Question-only KURE-v1 embedder used by the retrieval stage.

    It does not embed documents and has no access to Chroma mutation methods.
    """

    def __init__(
        self,
        model_name: str = "nlpai-lab/KURE-v1",
        device: str = "cpu",
        expected_dimension: int = 1024,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name,
            device=device,
            local_files_only=True,
        )
        actual_dimension = self.model.get_embedding_dimension()
        if actual_dimension != expected_dimension:
            raise ValueError(
                "Unexpected query embedding dimension: "
                f"expected {expected_dimension}, got {actual_dimension}"
            )

    def embed_query(self, query: str) -> Any:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        embeddings = self.model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings[0]


def load_read_only_collection(db_dir: Path, collection_name: str) -> tuple[Any, Any]:
    """Return the client and an existing collection without creating records."""
    if not db_dir.is_dir():
        raise FileNotFoundError(f"Chroma DB directory not found: {db_dir}")

    import chromadb

    client = chromadb.PersistentClient(path=str(db_dir))
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as exc:
        client.close()
        raise RuntimeError(f"Chroma collection not found: {collection_name}") from exc
    return client, collection


class Retriever:
    """Embed a question and query an existing Chroma collection."""

    def __init__(
        self,
        embedder: QueryEmbedder,
        collection: ReadOnlyCollection,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.collection = collection
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        query_id: str,
        query: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Return one project-contract retrieval result.

        Latency includes query embedding and Chroma lookup. Chroma's original
        distance is retained and ``similarity`` is derived for cosine space.
        """
        query_id = query_id.strip()
        query = query.strip()
        requested_k = self.config.evaluation_max_k if top_k is None else top_k

        if not query_id:
            raise ValueError("query_id must not be empty")
        if not query:
            raise ValueError("query must not be empty")
        if requested_k < 1:
            raise ValueError("top_k must be at least 1")

        record_count = self.collection.count()
        if record_count < 1:
            raise RuntimeError("The Chroma collection is empty")

        started = time.perf_counter()
        vector = self._as_vector_list(self.embedder.embed_query(query))
        raw = self.collection.query(
            query_embeddings=[vector],
            n_results=min(requested_k, record_count),
            include=["documents", "metadatas", "distances"],
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        return {
            "query_id": query_id,
            "query": query,
            "evaluation_max_k": requested_k,
            "retrieval_latency_ms": round(latency_ms, 3),
            "embedding_model": self.config.embedding_model,
            "index_version": self.config.index_version,
            "distance_metric": self.config.distance_metric,
            "results": self._normalize_results(raw),
        }

    @staticmethod
    def _as_vector_list(vector: Any) -> list[float]:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
            raise TypeError("embed_query() must return a one-dimensional vector")
        if len(vector) == 1 and isinstance(vector[0], Sequence):
            vector = vector[0]
        values = [float(value) for value in vector]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("query embedding must be non-empty and finite")
        return values

    @staticmethod
    def _first_row(raw: Mapping[str, Any], key: str) -> list[Any]:
        value = raw.get(key) or []
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"Invalid Chroma response field: {key}")
        if not value:
            return []
        first = value[0]
        if not isinstance(first, Sequence) or isinstance(first, (str, bytes)):
            raise ValueError(f"Invalid Chroma response row: {key}")
        return list(first)

    @classmethod
    def _normalize_results(cls, raw: Mapping[str, Any]) -> list[dict[str, Any]]:
        ids = cls._first_row(raw, "ids")
        documents = cls._first_row(raw, "documents")
        metadatas = cls._first_row(raw, "metadatas")
        distances = cls._first_row(raw, "distances")
        if len({len(ids), len(documents), len(metadatas), len(distances)}) != 1:
            raise ValueError("Chroma result fields have inconsistent lengths")

        results: list[dict[str, Any]] = []
        for rank, (chunk_id, text, metadata, raw_distance) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            if not isinstance(metadata, Mapping):
                metadata = {}
            distance = float(raw_distance)
            if not math.isfinite(distance):
                raise ValueError(f"Non-finite distance at rank {rank}")
            results.append(
                {
                    "rank": rank,
                    "chunk_id": str(chunk_id),
                    "document_id": metadata.get("document_id"),
                    "distance": distance,
                    "similarity": 1.0 - distance,
                    "text": text,
                    "metadata": dict(metadata),
                }
            )
        return results

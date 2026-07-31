# 질문
# → 질문 임베딩
# → FAISS 검색
# → Top-k Chunk 반환

import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from src.embedding import (
    embed_query,
    load_embedding_model
)


BASE_DIR = Path(__file__).resolve().parents[1]

FAISS_INDEX_PATH = (
    BASE_DIR / "indexes" / "faiss.index"
)

METADATA_PATH = (
    BASE_DIR / "indexes" / "metadata.json"
)


class Retriever:
    """
    임베딩 모델과 FAISS 인덱스를 이용해
    관련 Chunk를 검색한다.
    """

    def __init__(self) -> None:
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                "faiss.index가 없습니다. "
                "먼저 python -m src.vector_store를 실행하세요."
            )

        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                "metadata.json이 없습니다. "
                "먼저 python -m src.vector_store를 실행하세요."
            )

        self.embedding_model: SentenceTransformer = (
            load_embedding_model()
        )

        self.index = faiss.read_index(
            str(FAISS_INDEX_PATH)
        )

        with METADATA_PATH.open(
            "r",
            encoding="utf-8"
        ) as file:
            self.chunks: list[dict] = json.load(file)

    def retrieve(
        self,
        question: str,
        top_k: int = 3
    ) -> list[dict]:
        """
        질문과 유사한 Chunk를 Top-k개 검색한다.
        """

        if not question.strip():
            raise ValueError(
                "질문이 비어 있습니다."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k는 1 이상이어야 합니다."
            )

        query_embedding = embed_query(
            query=question,
            model=self.embedding_model
        )

        actual_top_k = min(
            top_k,
            self.index.ntotal
        )

        scores, indices = self.index.search(
            query_embedding,
            actual_top_k
        )

        results: list[dict] = []

        for rank, (score, metadata_index) in enumerate(
            zip(scores[0], indices[0]),
            start=1
        ):
            if metadata_index < 0:
                continue

            chunk = self.chunks[int(metadata_index)]

            results.append({
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "text": chunk["text"],
                "chunk_index": chunk["chunk_index"]
            })

        return results
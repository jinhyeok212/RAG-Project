"""SentenceTransformer 임베딩 모델 래퍼."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval_utils import load_environment


class KoreanEmbeddingModel:
    """환경변수로 선택한 한국어 문장 임베딩 모델을 관리합니다."""

    def __init__(self, model_name: str | None = None) -> None:
        load_environment()
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask"
        ).strip()
        if not self.model_name:
            raise ValueError("EMBEDDING_MODEL이 비어 있습니다.")
        # 모델 다운로드와 로드는 이 객체가 실제로 생성될 때만 일어납니다.
        self.model = SentenceTransformer(self.model_name)

    @property
    def dimension(self) -> int:
        """모델이 생성하는 벡터 차원을 반환합니다."""
        # sentence-transformers 신버전 API를 우선 사용하고 구버전도 호환합니다.
        if hasattr(self.model, "get_embedding_dimension"):
            dimension = self.model.get_embedding_dimension()
        else:
            dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("임베딩 차원을 확인할 수 없습니다.")
        return int(dimension)

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """문장들을 L2 정규화된 float32 벡터로 변환합니다."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=len(texts) > batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

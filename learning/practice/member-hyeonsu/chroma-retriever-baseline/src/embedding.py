"""문장 텍스트를 숫자 벡터(embedding)로 변환한다."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """문서와 질문에 반드시 같은 임베딩 모델을 적용하는 얇은 래퍼."""

    def __init__(self, model_name: str) -> None:
        # 첫 실행 때 Hugging Face에서 모델 파일을 내려받을 수 있다.
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """여러 문장을 한 번에 임베딩해 일반 Python list로 반환한다."""
        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,  # cosine 비교가 안정적이도록 길이를 1로 맞춘다.
            show_progress_bar=False,
        )
        return vectors.tolist()

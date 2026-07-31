import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


def load_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME
) -> SentenceTransformer:
    """
    Sentence Transformer 임베딩 모델을 불러온다.
    최초 실행 시 모델 파일이 자동 다운로드된다.
    """
    print(f"임베딩 모델 로딩: {model_name}")

    return SentenceTransformer(model_name)


def embed_documents(
    texts: list[str],
    model: SentenceTransformer
) -> np.ndarray:
    """
    여러 문서 Chunk를 임베딩한다.
    """
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return np.asarray(
        embeddings,
        dtype="float32"
    )


def embed_query(
    query: str,
    model: SentenceTransformer
) -> np.ndarray:
    """
    사용자 질문 하나를 임베딩한다.
    """
    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return np.asarray(
        embedding,
        dtype="float32"
    )
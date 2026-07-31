# documents.json 읽기
# → Chunking
# → Chunk 임베딩
# → FAISS 인덱스 생성
# → faiss.index 저장
# → metadata.json 저장

import json
from pathlib import Path

import faiss

from src.chunking import create_chunks
from src.embedding import (
    embed_documents,
    load_embedding_model
)


BASE_DIR = Path(__file__).resolve().parents[1]

DOCUMENTS_PATH = BASE_DIR / "data" / "documents.json"
INDEX_DIR = BASE_DIR / "indexes"

FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"


def load_json(file_path: Path) -> list[dict]:
    """
    JSON 파일을 불러온다.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"파일이 존재하지 않습니다: {file_path}"
        )

    with file_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def save_metadata(
    chunks: list[dict],
    file_path: Path
) -> None:
    """
    FAISS 벡터 순서와 대응되는 Chunk 정보를 저장한다.
    """
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with file_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )


def build_vector_store(
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> None:
    """
    문서를 Chunk로 나누고 FAISS 인덱스를 생성한다.
    """

    documents = load_json(DOCUMENTS_PATH)

    chunks = create_chunks(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    if not chunks:
        raise ValueError(
            "생성된 Chunk가 없습니다."
        )

    print(f"문서 수: {len(documents)}")
    print(f"Chunk 수: {len(chunks)}")

    embedding_model = load_embedding_model()

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    chunk_embeddings = embed_documents(
        texts=chunk_texts,
        model=embedding_model
    )

    dimension = chunk_embeddings.shape[1]

    # 정규화된 벡터의 내적을 이용해
    # 코사인 유사도 순위를 계산한다.
    index = faiss.IndexFlatIP(dimension)

    index.add(chunk_embeddings)

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(FAISS_INDEX_PATH)
    )

    save_metadata(
        chunks=chunks,
        file_path=METADATA_PATH
    )

    print(f"벡터 수: {index.ntotal}")
    print(f"FAISS 저장 위치: {FAISS_INDEX_PATH}")
    print(f"메타데이터 저장 위치: {METADATA_PATH}")


def main() -> None:
    build_vector_store(
        chunk_size=500,
        chunk_overlap=50
    )


if __name__ == "__main__":
    main()
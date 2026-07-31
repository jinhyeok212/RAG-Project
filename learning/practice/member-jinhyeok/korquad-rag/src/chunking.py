def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[str]:
    """
    텍스트를 문자 수 기준으로 분할한다.

    예:
    chunk_size=500
    chunk_overlap=50

    첫 번째 Chunk: 0~499
    두 번째 Chunk: 450~949
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size는 1 이상이어야 합니다."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap은 0 이상이어야 합니다."
        )

    if chunk_size <= chunk_overlap:
        raise ValueError(
            "chunk_size는 chunk_overlap보다 커야 합니다."
        )

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - chunk_overlap

    return chunks


def create_chunks(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[dict]:
    """
    documents.json의 문서를 Chunk 목록으로 변환한다.
    """

    chunks: list[dict] = []

    for document in documents:
        split_chunks = split_text(
            text=document["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for chunk_index, chunk_text in enumerate(split_chunks):
            chunks.append({
                "chunk_id": (
                    f"{document['doc_id']}_{chunk_index}"
                ),
                "doc_id": document["doc_id"],
                "title": document["title"],
                "text": chunk_text,
                "chunk_index": chunk_index
            })

    return chunks
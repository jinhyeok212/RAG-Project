"""샘플 context를 청킹하고 FAISS 인덱스를 만드는 단계."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_utils import (
    SAMPLE_DIR,
    VECTOR_DIR,
    chunk_text,
    e5_passage,
    ensure_project_dirs,
    read_jsonl,
    write_faiss_index,
    write_jsonl,
)


def import_embedding_dependencies():
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "필요한 패키지가 없습니다. 먼저 다음 명령을 실행하세요:\n"
            "pip install -r requirements.txt"
        ) from exc
    return faiss, np, SentenceTransformer


def build_chunks(documents: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    chunks = []
    for doc in documents:
        for chunk_idx, chunk in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}_chunk_{chunk_idx:03d}",
                    "doc_id": doc["doc_id"],
                    "chunk_idx": chunk_idx,
                    "text": chunk,
                    "title": doc.get("title", ""),
                    "source": doc.get("source", ""),
                }
            )
    return chunks


def build_index(
    documents_path: Path,
    output_dir: Path,
    model_name: str,
    chunk_size: int,
    overlap: int,
    batch_size: int,
) -> tuple[int, int]:
    faiss, np, SentenceTransformer = import_embedding_dependencies()

    documents = read_jsonl(documents_path)
    chunks = build_chunks(documents, chunk_size, overlap)
    if not chunks:
        raise ValueError("생성된 청크가 없습니다. documents.jsonl을 확인하세요.")

    model = SentenceTransformer(model_name)
    passages = [e5_passage(row["text"]) for row in chunks]
    embeddings = model.encode(
        passages,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.asarray(embeddings, dtype="float32"))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_faiss_index(faiss, index, output_dir / "index.faiss")
    write_jsonl(output_dir / "metadata.jsonl", chunks)

    config = {
        "model_name": model_name,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_dimension": int(dimension),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "similarity": "cosine similarity via normalized vectors + inner product",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return len(documents), len(chunks)


def main():
    parser = argparse.ArgumentParser(description="context 샘플을 청킹하고 FAISS 인덱스를 만듭니다.")
    parser.add_argument("--documents", type=Path, default=SAMPLE_DIR / "documents.jsonl")
    parser.add_argument("--model-name", default="intfloat/multilingual-e5-small")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    ensure_project_dirs()
    output_dir = args.output_dir or VECTOR_DIR / "faiss" / f"chunk_{args.chunk_size}"
    doc_count, chunk_count = build_index(
        documents_path=args.documents,
        output_dir=output_dir,
        model_name=args.model_name,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )

    print(f"문서 수: {doc_count}")
    print(f"청크 수: {chunk_count}")
    print(f"FAISS 인덱스 저장: {output_dir / 'index.faiss'}")
    print(f"메타데이터 저장: {output_dir / 'metadata.jsonl'}")


if __name__ == "__main__":
    main()

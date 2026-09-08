from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import chromadb
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data/processed/ontong_youth_chunks_v1/chunks.jsonl"
DEFAULT_INDEX_DIR = Path.home() / "chroma_indexes/ontong_youth_chroma_v1"
DEFAULT_COLLECTION = "ontong_youth_chunks_v1_sklearn_tfidf_svd"
KST = timezone(timedelta(hours=9))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_metadata(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False)


def make_chroma_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    source = {
        "chunk_id": chunk.get("chunk_id", ""),
        "document_id": chunk.get("document_id", ""),
        "chunk_index": chunk.get("chunk_index", 0),
        "start_char": chunk.get("start_char", 0),
        "end_char": chunk.get("end_char", 0),
        "char_length": chunk.get("char_length", 0),
        "title": chunk.get("title", ""),
        "source": chunk.get("source", ""),
        "url": chunk.get("url", ""),
    }
    metadata = chunk.get("metadata") or {}
    for key in [
        "chunking_version",
        "text_field",
        "chunk_size",
        "chunk_overlap",
        "source_policy_no",
        "primary_category",
        "primary_mclsf",
        "supervising_agency",
        "operating_agency",
        "application_period",
        "business_start_date",
        "business_end_date",
        "target_min_age",
        "target_max_age",
        "target_age_limit_yn",
        "income_condition_code",
        "region_zip_codes",
        "application_url",
        "reference_url_1",
        "normalization_status",
        "document_quality_score",
    ]:
        source[key] = metadata.get(key, "")
    return {key: clean_metadata(value) for key, value in source.items()}


def build_encoder(max_features: int, svd_dim: int, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 4),
                    max_features=max_features,
                    lowercase=True,
                ),
            ),
            ("svd", TruncatedSVD(n_components=svd_dim, random_state=random_state)),
            ("normalize", Normalizer(norm="l2")),
        ]
    )


def add_batches(collection, chunks: list[dict[str, Any]], embeddings, batch_size: int) -> None:
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        batch = chunks[start:end]
        collection.add(
            ids=[chunk["chunk_id"] for chunk in batch],
            documents=[chunk.get("text", "") for chunk in batch],
            embeddings=embeddings[start:end].tolist(),
            metadatas=[make_chroma_metadata(chunk) for chunk in batch],
        )


def run(
    *,
    chunks_path: Path,
    index_dir: Path,
    collection_name: str,
    version: str,
    workspace_pointer: Path | None,
    max_features: int,
    svd_dim: int,
    batch_size: int,
    random_state: int,
    reset_collection: bool,
) -> dict[str, Any]:
    chunks = read_jsonl(chunks_path)
    texts = [chunk.get("text", "") for chunk in chunks]
    effective_svd_dim = min(svd_dim, max(2, len(chunks) - 1))
    encoder = build_encoder(max_features, effective_svd_dim, random_state)

    start = time.perf_counter()
    embeddings = encoder.fit_transform(texts).astype("float32")
    embedding_elapsed_ms = (time.perf_counter() - start) * 1000

    index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    if reset_collection:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=collection_name,
        configuration={
            "hnsw": {
                "space": "cosine",
                "batch_size": min(batch_size, 100),
                "sync_threshold": min(batch_size, 100),
            }
        },
        metadata={
            "version": version,
            "embedding_backend": "sklearn_char_tfidf_svd",
        },
    )

    add_start = time.perf_counter()
    add_batches(collection, chunks, embeddings, batch_size)
    add_elapsed_ms = (time.perf_counter() - add_start) * 1000

    encoder_path = index_dir / "sklearn_tfidf_svd_encoder.joblib"
    pointer_path = workspace_pointer or ROOT / "data/indexes" / f"{index_dir.name}_pointer.json"
    joblib.dump(
        {
            "encoder": encoder,
            "collection_name": collection_name,
            "embedding_backend": "sklearn_char_tfidf_svd",
            "max_features": max_features,
            "svd_dim": effective_svd_dim,
            "random_state": random_state,
        },
        encoder_path,
    )

    manifest = {
        "version": version,
        "created_at": datetime.now(KST).isoformat(timespec="seconds"),
        "input": {
            "chunks_path": str(chunks_path),
        },
        "output": {
            "index_dir": str(index_dir),
            "chroma_path": str(index_dir / "chroma"),
            "encoder_path": str(encoder_path),
            "manifest_path": str(index_dir / "chroma_manifest.json"),
            "workspace_pointer_path": str(pointer_path),
        },
        "collection": {
            "name": collection_name,
            "count": collection.count(),
        },
        "embedding": {
            "backend": "sklearn_char_tfidf_svd",
            "tfidf_analyzer": "char_wb",
            "tfidf_ngram_range": [2, 4],
            "max_features": max_features,
            "svd_dim": effective_svd_dim,
            "normalization": "l2",
        },
        "timing": {
            "embedding_elapsed_ms": round(embedding_elapsed_ms, 3),
            "chroma_add_elapsed_ms": round(add_elapsed_ms, 3),
        },
        "counts": {
            "chunk_count": len(chunks),
            "embedding_count": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
        },
    }
    write_json(index_dir / "chroma_manifest.json", manifest)
    write_json(
        pointer_path,
        {
            "version": f"{version}_pointer",
            "reason": "Chroma persistent HNSW files failed under the Korean workspace path, so the runtime index is stored under an ASCII path.",
            "index_dir": str(index_dir),
            "chroma_path": str(index_dir / "chroma"),
            "encoder_path": str(encoder_path),
            "collection_name": collection_name,
            "manifest_path": str(index_dir / "chroma_manifest.json"),
            "created_at": manifest["created_at"],
        },
    )
    client.close()
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Ontong Youth Chroma index v1.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--version", default="ontong_youth_chroma_v1")
    parser.add_argument("--workspace-pointer", type=Path, default=None)
    parser.add_argument("--max-features", type=int, default=80000)
    parser.add_argument("--svd-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-reset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run(
        chunks_path=args.chunks,
        index_dir=args.index_dir,
        collection_name=args.collection,
        version=args.version,
        workspace_pointer=args.workspace_pointer,
        max_features=args.max_features,
        svd_dim=args.svd_dim,
        batch_size=args.batch_size,
        random_state=args.random_state,
        reset_collection=not args.no_reset,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

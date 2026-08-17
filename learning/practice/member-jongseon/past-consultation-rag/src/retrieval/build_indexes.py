from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.common import ensure_dirs, load_config
from src.retrieval.ann_index import DenseANNRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseExactRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    processed = Path(cfg["paths"]["processed_dir"])
    index_dir = Path(cfg["paths"]["index_dir"])
    ensure_dirs(index_dir)
    df = pd.read_parquet(processed / "train_rag_corpus.parquet")
    doc_ids = df["doc_id"].tolist()
    texts = df["retrieval_text"].fillna("").tolist()

    bm25_cfg = cfg["retrieval"]["bm25"]
    bm25 = BM25Retriever(k1=float(bm25_cfg["k1"]), b=float(bm25_cfg["b"]), tokenizer=bm25_cfg.get("tokenizer", "mixed"))
    bm25.fit(doc_ids, texts)
    bm25.save(index_dir / "bm25.pkl")

    dense_cfg = cfg["retrieval"]["dense"]
    dense = DenseExactRetriever(
        max_features=int(dense_cfg.get("max_features", 80000)),
        char_ngram_min=int(dense_cfg.get("char_ngram_min", 2)),
        char_ngram_max=int(dense_cfg.get("char_ngram_max", 4)),
    )
    dense.fit(doc_ids, texts)
    dense.save(index_dir / "dense_exact.pkl")

    if cfg["retrieval"].get("ann", {}).get("enabled", True):
        ann_cfg = cfg["retrieval"]["ann"]
        ann = DenseANNRetriever(dense, svd_dim=int(ann_cfg.get("svd_dim", 128)), n_neighbors=int(ann_cfg.get("n_neighbors", 50)))
        ann.fit()
        ann.save(index_dir / "dense_ann.pkl")

    docs = {row["doc_id"]: row for row in df.to_dict("records")}
    joblib.dump(docs, index_dir / "docs.joblib")
    print(f"Indexed {len(doc_ids):,} documents into {index_dir}")


if __name__ == "__main__":
    main()

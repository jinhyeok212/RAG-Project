from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.common import compact_text, ensure_dirs, load_config, stable_id, write_jsonl
from src.data.privacy_masking import detect_privacy_patterns, mask_text


def find_category_file(raw_root: Path, split: str, category: str) -> Path:
    suffix = "train" if split == "Training" else "validation"
    path = raw_root / split / f"{category}_{suffix}.csv"
    if not path.exists():
        candidates = sorted((raw_root / split).glob(f"*_{suffix}.csv"))
        if not candidates:
            raise FileNotFoundError(f"No CSV files in {raw_root / split}")
        return candidates[0]
    return path


def row_value(row: pd.Series, col: str | None) -> str:
    return compact_text(row.get(col, "")) if col else ""


def extract_entities(row: pd.Series, entity_cols: list[str]) -> list[dict[str, str]]:
    entities = []
    for col in entity_cols:
        value = compact_text(row.get(col, ""))
        if value:
            entities.append({"type": col, "value": mask_text(value)})
    return entities


def compose_retrieval_text(doc: dict[str, Any], version: str) -> str:
    parts = []
    if version in {"C", "D"}:
        if doc.get("category"):
            parts.append(f"카테고리: {doc['category']}")
        if doc.get("intent"):
            parts.append(f"인텐트: {doc['intent']}")
    if version in {"B", "C", "D"} and doc.get("previous_context_original"):
        parts.append(f"이전 문맥: {doc['previous_context_original']}")
    parts.append(f"고객 질문: {doc.get('question_original', '')}")
    if version == "D":
        parts.append(f"상담사 답변: {doc.get('answer_original', '')}")
    return "\n".join(p for p in parts if compact_text(p))


def reconstruct(path: Path, split_name: str, cfg: dict, max_docs: int | None) -> list[dict[str, Any]]:
    cols = cfg["columns"]
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in [cols["consultation_id"], cols["turn_order"], cols["qa_id"], cols["qa_role"], cols["speaker"], cols["text"]]:
        if col not in df.columns:
            raise ValueError(f"Required column `{col}` not found in {path}")

    df["_turn_order_num"] = pd.to_numeric(df[cols["turn_order"]], errors="coerce").fillna(0)
    df = df.sort_values([cols["consultation_id"], "_turn_order_num", "IDX" if "IDX" in df.columns else cols["turn_order"]])

    docs = []
    prev_turns = int(cfg["data"].get("previous_context_turns", 3))
    version = cfg["data"].get("retrieval_text_version", "C")
    entity_cols = [c for c in cfg["columns"].get("entities", []) if c in df.columns]

    grouped = df.groupby(cols["consultation_id"], sort=False)
    for consultation_id, group in grouped:
        history: list[str] = []
        pending_by_qa: dict[str, list[tuple[pd.Series, list[str]]]] = defaultdict(list)
        for _, row in group.iterrows():
            text_original = mask_text(row_value(row, cols["text"]))
            qa_role = row_value(row, cols["qa_role"]).lower()
            speaker = row_value(row, cols["speaker"]).lower()
            qa_id = row_value(row, cols["qa_id"])
            turn = row_value(row, cols["turn_order"])

            is_question = qa_role == "q" or speaker == "c"
            is_answer = qa_role == "a" or speaker == "s"
            if is_question:
                pending_by_qa[qa_id].append((row, history[-prev_turns:].copy()))
            elif is_answer:
                queue = pending_by_qa.get(qa_id) or []
                if queue:
                    qrow, ctx = queue.pop(0)
                    q_text = mask_text(row_value(qrow, cols["text"]))
                    a_text = text_original
                    quality_flags = []
                    if not q_text:
                        quality_flags.append("missing_question")
                    if not a_text:
                        quality_flags.append("missing_answer")
                    privacy = sorted(set(detect_privacy_patterns(q_text) + detect_privacy_patterns(a_text)))
                    if privacy:
                        quality_flags.append("privacy_masked")
                    doc = {
                        "doc_id": stable_id(split_name, consultation_id, qa_id, row_value(qrow, cols["turn_order"]), turn, q_text, a_text),
                        "source_split": split_name.lower(),
                        "category": row_value(qrow, cols.get("category")),
                        "intent": row_value(qrow, cols.get("intent")),
                        "consultation_id": consultation_id,
                        "qa_id": qa_id,
                        "question_original": q_text,
                        "answer_original": a_text,
                        "previous_context_original": " ".join(ctx),
                        "question_normalized": compact_text(q_text).lower(),
                        "answer_normalized": compact_text(a_text).lower(),
                        "retrieval_text": "",
                        "entities": extract_entities(qrow, entity_cols) + extract_entities(row, entity_cols),
                        "quality_flags": quality_flags,
                        "metadata": {
                            "question_turn_order": row_value(qrow, cols["turn_order"]),
                            "answer_turn_order": turn,
                            "question_speaker": row_value(qrow, cols["speaker"]),
                            "answer_speaker": row_value(row, cols["speaker"]),
                        },
                    }
                    doc["retrieval_text"] = compose_retrieval_text(doc, version)
                    docs.append(doc)
                    if max_docs and len(docs) >= max_docs:
                        return docs
            history.append(f"{speaker}: {text_original}")
    return docs


def save_outputs(train_docs: list[dict], val_docs: list[dict], cfg: dict) -> None:
    processed = Path(cfg["paths"]["processed_dir"])
    reports = Path(cfg["paths"]["report_dir"])
    ensure_dirs(processed, reports)
    train_df = pd.DataFrame(train_docs)
    val_df = pd.DataFrame(val_docs)
    train_df.to_parquet(processed / "train_rag_corpus.parquet", index=False)
    val_df.to_parquet(processed / "validation_rag_corpus.parquet", index=False)
    write_jsonl(processed / "train_rag_corpus_sample.jsonl", train_docs[:100])
    positives_by_cat_intent: dict[tuple[str, str], list[str]] = defaultdict(list)
    positives_by_intent: dict[str, list[str]] = defaultdict(list)
    positives_by_category: dict[str, list[str]] = defaultdict(list)
    for doc in train_docs:
        category = doc.get("category") or ""
        intent = doc.get("intent") or ""
        positives_by_cat_intent[(category, intent)].append(doc["doc_id"])
        if intent:
            positives_by_intent[intent].append(doc["doc_id"])
        if category:
            positives_by_category[category].append(doc["doc_id"])

    queries = []
    for doc in val_docs:
        if not doc.get("question_original") or not doc.get("answer_original"):
            continue
        category = doc.get("category") or ""
        intent = doc.get("intent") or ""
        positive_doc_ids = positives_by_cat_intent.get((category, intent), [])
        match_rule = "category_intent"
        if not positive_doc_ids and intent:
            positive_doc_ids = positives_by_intent.get(intent, [])
            match_rule = "intent"
        if not positive_doc_ids and category:
            positive_doc_ids = positives_by_category.get(category, [])
            match_rule = "category"
        queries.append(
            {
                "query_id": doc["doc_id"],
                "query": doc["question_original"],
                "positive_doc_ids": positive_doc_ids[:100],
                "category": category,
                "intent": intent,
                "consultation_id": doc.get("consultation_id"),
                "metadata": {
                    "qa_id": doc.get("qa_id"),
                    "positive_match_rule": match_rule,
                    "validation_answer_doc_id": doc["doc_id"],
                },
            }
        )
    write_jsonl(processed / "validation_retrieval_queries.jsonl", queries)
    md = [
        "# 데이터 품질 리포트",
        "",
        f"- Train QA docs: {len(train_docs):,}",
        f"- Validation QA docs / queries: {len(val_docs):,}",
        f"- Retrieval text version: {cfg['data'].get('retrieval_text_version', 'C')}",
        "",
        "## Quality flags",
    ]
    for name, docs in [("train", train_docs), ("validation", val_docs)]:
        counts: dict[str, int] = defaultdict(int)
        for doc in docs:
            for flag in doc.get("quality_flags", []):
                counts[flag] += 1
        md.append(f"### {name}")
        if counts:
            for flag, count in sorted(counts.items()):
                md.append(f"- {flag}: {count:,}")
        else:
            md.append("- 주요 결측/개인정보 flag 없음")
    (reports / "data_quality_report.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    raw_root = Path(cfg["paths"]["raw_root"])
    category = cfg["data"]["mvp_category"]
    train_path = find_category_file(raw_root, "Training", category)
    val_path = find_category_file(raw_root, "Validation", category)
    train_docs = reconstruct(train_path, "train", cfg, int(cfg["data"].get("max_train_docs", 20000)))
    val_docs = reconstruct(val_path, "validation", cfg, int(cfg["data"].get("max_validation_queries", 2000)))
    save_outputs(train_docs, val_docs, cfg)


if __name__ == "__main__":
    main()

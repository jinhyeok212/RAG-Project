from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import chromadb
import joblib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data/processed/ontong_youth_chunks_v1/chunks.jsonl"
DEFAULT_EVAL = ROOT / "data/processed/ontong_youth_chunks_v1/eval_questions_with_gold_chunks.jsonl"
DEFAULT_POINTER = ROOT / "data/indexes/ontong_youth_chroma_v1_pointer.json"
DEFAULT_OUTPUT_DIR = ROOT / "data/processed/ontong_youth_rag_mvp_v1"
KST = timezone(timedelta(hours=9))


FIELD_LABELS = {
    "plcySprtCn": ["지원 내용"],
    "ptcpPrpTrgtCn": ["참여 대상"],
    "addAplyQlfcCndCn": ["추가 신청 자격"],
    "aplyYmd": ["신청 기간"],
    "plcyAplyMthdCn": ["신청 방법"],
    "srngMthdCn": ["심사 방법"],
    "sbmsnDcmntCn": ["제출 서류"],
    "etcMttrCn": ["기타 사항"],
}

QUESTION_TYPE_LABELS = {
    "eligibility": ["참여 대상", "추가 신청 자격", "지원 내용"],
    "application_period": ["신청 기간"],
    "application_method": ["신청 방법"],
    "support_content": ["지원 내용"],
    "required_documents": ["제출 서류"],
    "specific_policy_search": ["정책명", "정책 요약", "지원 내용"],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_pointer(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[가-힣A-Za-z0-9]+", (text or "").lower())
    tokens = set(words)
    compact = "".join(words)
    for n in (2, 3):
        tokens.update(compact[i : i + n] for i in range(max(0, len(compact) - n + 1)))
    return {token for token in tokens if token}


def labels_for_question(question: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    answer_source_field = question.get("answer_source_field", "")
    for field, field_labels in FIELD_LABELS.items():
        if field in answer_source_field:
            labels.extend(field_labels)
    labels.extend(QUESTION_TYPE_LABELS.get(question.get("question_type", ""), []))
    if not labels:
        labels = ["정책명", "정책 요약", "지원 내용", "참여 대상", "신청 기간", "신청 방법"]
    deduped = []
    for label in labels:
        if label not in deduped:
            deduped.append(label)
    return deduped


def split_evidence_units(text: str) -> list[str]:
    units = []
    current = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                units.append(" ".join(current).strip())
                current = []
            continue
        current.append(line)
        if len(" ".join(current)) >= 260:
            units.append(" ".join(current).strip())
            current = []
    if current:
        units.append(" ".join(current).strip())
    return [unit for unit in units if unit]


def score_unit(unit: str, query_tokens: set[str], labels: list[str]) -> float:
    unit_tokens = tokenize(unit)
    overlap = len(query_tokens & unit_tokens)
    label_bonus = sum(8 for label in labels if label in unit)
    date_bonus = 2 if re.search(r"\d{4}[.-]?\d{2}[.-]?\d{2}|\d{8}|\d+\.\d+", unit) else 0
    amount_bonus = 2 if re.search(r"\d+\s*(만원|천원|원|%)", unit) else 0
    return overlap + label_bonus + date_bonus + amount_bonus + min(len(unit), 300) / 1000


def select_document(top_results: list[dict[str, Any]]) -> str:
    return top_results[0]["document_id"] if top_results else ""


def generate_grounded_answer(
    *,
    question: dict[str, Any],
    selected_document_id: str,
    selected_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    if not selected_chunks:
        return {
            "answer": "검색된 근거만으로는 답변을 만들기 어렵습니다.",
            "evidence_snippets": [],
            "selected_title": "",
            "selected_url": "",
        }

    query_tokens = tokenize(question["query"])
    labels = labels_for_question(question)
    units = []
    for chunk in sorted(selected_chunks, key=lambda row: row["chunk_index"]):
        for unit in split_evidence_units(chunk.get("text", "")):
            units.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": unit,
                    "score": score_unit(unit, query_tokens, labels),
                }
            )
    units.sort(key=lambda row: row["score"], reverse=True)

    evidence = []
    used_texts = set()
    total_chars = 0
    for unit in units:
        normalized = unit["text"]
        if normalized in used_texts:
            continue
        if total_chars + len(normalized) > 900 and evidence:
            continue
        evidence.append({"chunk_id": unit["chunk_id"], "text": normalized})
        used_texts.add(normalized)
        total_chars += len(normalized)
        if len(evidence) >= 3:
            break

    title = selected_chunks[0].get("title", "")
    url = selected_chunks[0].get("url", "")
    if evidence:
        joined = " ".join(item["text"] for item in evidence)
        answer = f"{title} 기준으로 확인하면, {joined}"
    else:
        answer = f"{title} 문서가 검색되었지만, 질문에 직접 답할 근거 문장을 충분히 추출하지 못했습니다."

    return {
        "answer": answer,
        "evidence_snippets": evidence,
        "selected_title": title,
        "selected_url": url,
        "selected_document_id": selected_document_id,
    }


def compact_results(
    ids: list[str],
    distances: list[float],
    chunks_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    for rank, (chunk_id, distance) in enumerate(zip(ids, distances), start=1):
        chunk = chunks_by_id[chunk_id]
        results.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "document_id": chunk["document_id"],
                "chunk_index": chunk["chunk_index"],
                "distance": float(distance),
                "score": 1.0 - float(distance),
                "title": chunk.get("title", ""),
                "url": chunk.get("url", ""),
            }
        )
    return results


def hit_at_k(results: list[str], positives: set[str], k: int) -> float:
    return 1.0 if set(results[:k]) & positives else 0.0


def run(
    *,
    chunks_path: Path,
    eval_path: Path,
    pointer_path: Path,
    output_dir: Path,
    top_k: int,
    context_chunks_per_doc: int,
) -> dict[str, Any]:
    pointer = load_pointer(pointer_path)
    chunks = read_jsonl(chunks_path)
    eval_questions = read_jsonl(eval_path)
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

    encoder_bundle = joblib.load(pointer["encoder_path"])
    encoder = encoder_bundle["encoder"]
    client = chromadb.PersistentClient(path=pointer["chroma_path"])
    collection = client.get_collection(pointer["collection_name"])

    answer_rows = []
    metric_rows = []
    caution_rows = []

    for question in eval_questions:
        started = time.perf_counter()
        query_embedding = encoder.transform([question["query"]]).astype("float32").tolist()[0]
        response = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["distances", "documents", "metadatas"],
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        top_results = compact_results(response["ids"][0], response["distances"][0], chunks_by_id)
        selected_document_id = select_document(top_results)
        selected_chunks = [
            chunks_by_id[row["chunk_id"]]
            for row in top_results
            if row["document_id"] == selected_document_id
        ][:context_chunks_per_doc]
        generated = generate_grounded_answer(
            question=question,
            selected_document_id=selected_document_id,
            selected_chunks=selected_chunks,
        )

        gold_chunk_ids = set(question.get("gold_chunk_ids") or [])
        gold_document_ids = set(question.get("ground_truth_document_ids") or [])
        ranked_chunk_ids = [row["chunk_id"] for row in top_results]
        ranked_document_ids = [row["document_id"] for row in top_results]
        top1_doc_hit = hit_at_k(ranked_document_ids, gold_document_ids, 1)
        top5_doc_hit = hit_at_k(ranked_document_ids, gold_document_ids, 5)
        selected_doc_hit = 1.0 if selected_document_id in gold_document_ids else 0.0

        answer_row = {
            "query_id": question["query_id"],
            "query": question["query"],
            "question_type": question.get("question_type", ""),
            "answer_source_field": question.get("answer_source_field", ""),
            "ground_truth_document_ids": list(gold_document_ids),
            "gold_chunk_ids": list(gold_chunk_ids),
            "selected_document_id": selected_document_id,
            "selected_title": generated["selected_title"],
            "selected_url": generated["selected_url"],
            "answer": generated["answer"],
            "evidence_snippets": generated["evidence_snippets"],
            "top_results": top_results,
            "top1_document_hit": top1_doc_hit,
            "top5_document_hit": top5_doc_hit,
            "selected_document_hit": selected_doc_hit,
            "latency_ms": elapsed_ms,
        }
        metric_rows.append(
            {
                "query_id": question["query_id"],
                "question_type": question.get("question_type", ""),
                "top1_document_hit": top1_doc_hit,
                "top5_document_hit": top5_doc_hit,
                "selected_document_hit": selected_doc_hit,
                "answer_char_length": len(generated["answer"]),
                "evidence_count": len(generated["evidence_snippets"]),
                "latency_ms": elapsed_ms,
            }
        )
        answer_rows.append(answer_row)
        if selected_doc_hit == 0.0 or top5_doc_hit == 0.0:
            caution_rows.append(answer_row)

    summary = {
        "query_count": len(eval_questions),
        "top1_document_hit": round(mean(row["top1_document_hit"] for row in metric_rows), 6),
        "top5_document_hit": round(mean(row["top5_document_hit"] for row in metric_rows), 6),
        "selected_document_hit": round(mean(row["selected_document_hit"] for row in metric_rows), 6),
        "avg_answer_char_length": round(mean(row["answer_char_length"] for row in metric_rows), 3),
        "avg_evidence_count": round(mean(row["evidence_count"] for row in metric_rows), 3),
        "avg_latency_ms": round(mean(row["latency_ms"] for row in metric_rows), 3),
        "caution_case_count": len(caution_rows),
    }

    manifest = {
        "version": "ontong_youth_rag_mvp_v1",
        "created_at": datetime.now(KST).isoformat(timespec="seconds"),
        "input": {
            "chunks_path": str(chunks_path),
            "eval_questions_with_gold_chunks_path": str(eval_path),
            "chroma_pointer_path": str(pointer_path),
        },
        "output": {
            "output_dir": str(output_dir),
            "answers_path": str(output_dir / "rag_answers.jsonl"),
            "metrics_path": str(output_dir / "rag_answer_metrics.csv"),
            "caution_cases_path": str(output_dir / "rag_caution_cases.jsonl"),
            "report_path": str(output_dir / "rag_mvp_report.md"),
        },
        "retrieval": {
            "backend": "Chroma",
            "embedding_backend": encoder_bundle.get("embedding_backend", ""),
            "collection_name": pointer["collection_name"],
            "top_k": top_k,
            "context_chunks_per_doc": context_chunks_per_doc,
        },
        "generation": {
            "backend": "local_extractive_template",
            "policy": "Use only retrieved chunk text. No external LLM call.",
        },
        "summary": summary,
    }

    report = [
        "# Ontong Youth RAG MVP v1",
        "",
        f"- created_at: {manifest['created_at']}",
        f"- retrieval: Chroma / {pointer['collection_name']}",
        f"- generation: local extractive template",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in summary.items():
        report.append(f"| {key} | {value} |")
    report.extend(
        [
            "",
            "## Notes",
            "",
            "- selected_document_hit checks whether the document chosen for answer generation is one of the qrel-positive documents.",
            "- Answers are extractive MVP drafts, not final natural-language LLM answers.",
            "- Caution cases should be reviewed before wiring this into a user-facing chatbot.",
        ]
    )

    write_jsonl(output_dir / "rag_answers.jsonl", answer_rows)
    write_csv(output_dir / "rag_answer_metrics.csv", metric_rows)
    write_jsonl(output_dir / "rag_caution_cases.jsonl", caution_rows)
    write_json(output_dir / "rag_mvp_manifest.json", manifest)
    (output_dir / "rag_mvp_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    client.close()
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ontong Youth RAG MVP v1.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--context-chunks-per-doc", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run(
        chunks_path=args.chunks,
        eval_path=args.eval,
        pointer_path=args.pointer,
        output_dir=args.output_dir,
        top_k=args.top_k,
        context_chunks_per_doc=args.context_chunks_per_doc,
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

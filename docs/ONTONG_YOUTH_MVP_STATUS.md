# Ontong Youth RAG MVP Status

This document summarizes the project artifacts added from the Ontong Youth policy RAG MVP work.

## Current Scope

The included artifacts cover the following stages:

1. Ontong Youth policy API collection
2. Category and metadata normalization
3. MVP corpus selection
4. RAG document schema construction
5. Evaluation question and qrels construction
6. Fixed-size chunking experiments
7. Section-aware chunking experiments
8. Chroma-based retrieval experiments using a local sklearn char TF-IDF + SVD encoder
9. Local extractive answer-generation MVP evaluation

The repository does not include API keys, original KorQuAD large files, or binary vector indexes.

## Key Data Artifacts

| Purpose | Path |
| --- | --- |
| MVP RAG documents | `data/processed/ontong_youth_mvp_400_documents/documents.jsonl` |
| Document schema | `data/processed/ontong_youth_mvp_400_documents/DOCUMENT_SCHEMA.md` |
| Final evaluation questions | `data/processed/ontong_youth_eval_questions_v2_final_47/eval_questions.jsonl` |
| Document-level qrels | `data/processed/ontong_youth_eval_questions_v2_final_47/qrels.jsonl` |
| Fixed 500/50 chunks | `data/processed/ontong_youth_chunks_v1/chunks.jsonl` |
| Section-aware chunks | `data/processed/ontong_youth_chunks_section_v2/chunks.jsonl` |
| Chunking comparison report | `data/processed/ontong_youth_chunking_comparison_v1/chunking_comparison_report.md` |
| Section-vs-fixed comparison report | `data/processed/ontong_youth_chunking_section_comparison_v2/section_vs_fixed_report.md` |
| Fixed chunk RAG report | `data/processed/ontong_youth_rag_mvp_v1/rag_mvp_report.md` |
| Section chunk RAG report | `data/processed/ontong_youth_rag_section_v2/rag_mvp_report.md` |
| Human-readable PDF report | `data/processed/ontong_youth_project_report/ontong_youth_rag_mvp_report_readable_v3.pdf` |

## Current Counts

| Artifact | Count |
| --- | ---: |
| Collected Ontong Youth policy records | 2,728 |
| MVP documents | 400 |
| Final evaluation questions | 47 |
| qrels rows | 50 |
| Fixed 500/50 chunks | 1,060 |
| Section-aware chunks | 5,662 |

The qrels row count is larger than the question count because three questions have two valid answer documents.

## Retrieval Results

The current experiments use Chroma with a local `sklearn_char_tfidf_svd` encoder. They are baseline retrieval experiments, not Hugging Face embedding model comparisons.

| Setup | Top1 document hit | Top5 document hit | MRR / selected hit note |
| --- | ---: | ---: | --- |
| Fixed 500/50 Chroma SVD | 0.6596 | 0.8936 | broader candidate recall |
| Section-aware v2 Chroma SVD | 0.7234 | 0.8085 | stronger first-rank precision |

The current recommendation is not to fully replace fixed 500/50. Use fixed 500/50 as a broad candidate retrieval baseline and section-aware chunks as an answer-evidence or reranking candidate.

## Important Limitations

- No API key is included.
- Large binary indexes are not included. Rebuild them from the scripts if needed.
- BGE-M3, KURE-v1, Qwen3-Embedding, or other Hugging Face embedding models have not yet been benchmarked in these artifacts.
- The answer-generation MVP is a local extractive template, not a final LLM chatbot.
- qrels are document-level and gold chunk mapping is currently broad: chunks from qrel-positive documents are treated as gold candidates.

## Main Scripts

| Stage | Script |
| --- | --- |
| API collection | `scripts/fetch_ontong_youth_policy_full.py` |
| Missing page recovery | `scripts/recover_ontong_youth_policy_missing_pages.py` |
| Normalization | `scripts/normalize_ontong_youth_catalog.py` |
| MVP 400 selection | `scripts/select_ontong_youth_mvp_400.py` |
| Document schema build | `scripts/build_ontong_youth_mvp_documents.py` |
| Evaluation question generation | `scripts/generate_ontong_youth_eval_question_candidates_v2_refined.py` |
| Label-2 repair | `scripts/repair_ontong_youth_eval_label2.py` |
| Final eval/qrels build | `scripts/build_ontong_youth_eval_final_47.py` |
| Fixed chunking | `scripts/build_ontong_youth_chunks_v1.py` |
| Section-aware chunking | `scripts/build_ontong_youth_section_chunks_v2.py` |
| Chroma build | `scripts/build_ontong_youth_chroma_v1.py` |
| Chunk retrieval evaluation | `scripts/evaluate_ontong_youth_chunk_retrieval_v1.py` |
| RAG MVP run | `scripts/run_ontong_youth_rag_mvp_v1.py` |


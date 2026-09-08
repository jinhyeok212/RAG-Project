# Ontong Youth RAG MVP v1

- created_at: 2026-08-31T22:56:22+09:00
- retrieval: Chroma / ontong_youth_chunks_section_v2_sklearn_tfidf_svd
- generation: local extractive template

| metric | value |
| --- | ---: |
| query_count | 47 |
| top1_document_hit | 0.723404 |
| top5_document_hit | 0.808511 |
| selected_document_hit | 0.723404 |
| avg_answer_char_length | 240 |
| avg_evidence_count | 2.936 |
| avg_latency_ms | 76.482 |
| caution_case_count | 13 |

## Notes

- selected_document_hit checks whether the document chosen for answer generation is one of the qrel-positive documents.
- Answers are extractive MVP drafts, not final natural-language LLM answers.
- Caution cases should be reviewed before wiring this into a user-facing chatbot.

# Ontong Youth RAG MVP v1

- created_at: 2026-08-31T21:35:33+09:00
- retrieval: Chroma / ontong_youth_chunks_v1_sklearn_tfidf_svd
- generation: local extractive template

| metric | value |
| --- | ---: |
| query_count | 47 |
| top1_document_hit | 0.659574 |
| top5_document_hit | 0.893617 |
| selected_document_hit | 0.659574 |
| avg_answer_char_length | 374.809 |
| avg_evidence_count | 3 |
| avg_latency_ms | 73.222 |
| caution_case_count | 16 |

## Notes

- selected_document_hit checks whether the document chosen for answer generation is one of the qrel-positive documents.
- Answers are extractive MVP drafts, not final natural-language LLM answers.
- Caution cases should be reviewed before wiring this into a user-facing chatbot.

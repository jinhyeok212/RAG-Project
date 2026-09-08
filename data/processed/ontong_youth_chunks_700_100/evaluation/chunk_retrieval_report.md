# Ontong Youth Chunk Retrieval Evaluation v1

- created_at: 2026-08-31T21:14:24+09:00
- chunk_count: 797
- eval_question_count: 47
- gold mapping: v1 broad candidate mapping: all chunks from each qrel-positive document are gold.

| method | chunk Hit@1 | chunk Hit@3 | chunk Hit@5 | chunk MRR | document Hit@1 | document Hit@5 | document MRR | latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.7021 | 0.8085 | 0.8511 | 0.7706 | 0.7021 | 0.8511 | 0.7706 | 2.23 |
| Char TF-IDF | 0.6596 | 0.8298 | 0.9362 | 0.7631 | 0.6596 | 0.9362 | 0.7631 | 2.53 |
| Hybrid RRF | 0.6809 | 0.8511 | 0.9149 | 0.7773 | 0.6809 | 0.9149 | 0.7773 | 6.13 |

## Notes

- chunk Hit@k checks whether any retrieved chunk_id is in gold_chunk_ids.
- document Hit@k checks whether any retrieved chunk's document_id is in ground_truth_document_ids.
- v1 gold_chunk_ids are broad candidates: every chunk from a qrel-positive document is counted as gold.

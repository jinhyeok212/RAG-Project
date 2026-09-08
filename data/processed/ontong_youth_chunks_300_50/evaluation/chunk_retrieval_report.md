# Ontong Youth Chunk Retrieval Evaluation v1

- created_at: 2026-08-31T21:14:25+09:00
- chunk_count: 1742
- eval_question_count: 47
- gold mapping: v1 broad candidate mapping: all chunks from each qrel-positive document are gold.

| method | chunk Hit@1 | chunk Hit@3 | chunk Hit@5 | chunk MRR | document Hit@1 | document Hit@5 | document MRR | latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.7021 | 0.7872 | 0.8298 | 0.7635 | 0.7021 | 0.8298 | 0.7635 | 3.51 |
| Char TF-IDF | 0.7234 | 0.8298 | 0.8723 | 0.7830 | 0.7234 | 0.8723 | 0.7830 | 3.27 |
| Hybrid RRF | 0.6809 | 0.8511 | 0.9362 | 0.7745 | 0.6809 | 0.9362 | 0.7745 | 9.67 |

## Notes

- chunk Hit@k checks whether any retrieved chunk_id is in gold_chunk_ids.
- document Hit@k checks whether any retrieved chunk's document_id is in ground_truth_document_ids.
- v1 gold_chunk_ids are broad candidates: every chunk from a qrel-positive document is counted as gold.

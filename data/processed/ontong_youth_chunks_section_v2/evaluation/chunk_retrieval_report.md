# Ontong Youth Chunk Retrieval Evaluation v1

- created_at: 2026-08-31T22:55:18+09:00
- chunk_count: 5662
- eval_question_count: 47
- gold mapping: v1 broad candidate mapping: all chunks from each qrel-positive document are gold.

| method | chunk Hit@1 | chunk Hit@3 | chunk Hit@5 | chunk MRR | document Hit@1 | document Hit@5 | document MRR | latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.6809 | 0.8085 | 0.8723 | 0.7479 | 0.6809 | 0.8723 | 0.7479 | 5.66 |
| Char TF-IDF | 0.7447 | 0.8085 | 0.8298 | 0.7880 | 0.7447 | 0.8298 | 0.7880 | 2.70 |
| Hybrid RRF | 0.7872 | 0.8298 | 0.8511 | 0.8224 | 0.7872 | 0.8511 | 0.8224 | 9.41 |

## Notes

- chunk Hit@k checks whether any retrieved chunk_id is in gold_chunk_ids.
- document Hit@k checks whether any retrieved chunk's document_id is in ground_truth_document_ids.
- v1 gold_chunk_ids are broad candidates: every chunk from a qrel-positive document is counted as gold.

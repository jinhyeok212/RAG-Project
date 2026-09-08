# Section-aware Chunking v2 Comparison

created_at: 2026-08-31T22:57:05+09:00

## Retrieval Results

| chunking | retrieval | chunks | avg len | Hit@1 | Hit@5 | Hit@10 | Recall@5 | MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed 500/50 | Hybrid RRF | 1060 | 413.7 | 0.7021 | 0.9149 | 0.9574 | 0.5105 | 0.7812 |
| Fixed 500/50 | Chroma SVD | 1060 | 413.7 | 0.6596 | 0.8936 | 1.0000 | 0.5226 | 0.7525 |
| Section-aware v2 | Hybrid RRF | 5662 | 95.3 | 0.7872 | 0.8511 | 0.9149 | 0.1898 | 0.8224 |
| Section-aware v2 | Chroma SVD | 5662 | 95.3 | 0.7234 | 0.8085 | 0.8298 | 0.1829 | 0.7592 |

## RAG Selection Results

| chunking | Top1 doc hit | Top5 doc hit | selected doc hit | caution cases | avg answer chars |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 500/50 | 0.6596 | 0.8936 | 0.6596 | 16 | 374.8 |
| Section-aware v2 | 0.7234 | 0.8085 | 0.7234 | 13 | 240.0 |

## Interpretation

- Section-aware v2 improves Chroma Top1 from 0.6596 to 0.7234 and RAG selected_document_hit from 0.6596 to 0.7234.
- Fixed 500/50 keeps stronger candidate recall: Chroma Hit@5 0.8936 vs 0.8085, Hit@10 1.0000 vs 0.8298.
- Section-aware v2 creates many short chunks, so it is better for precise first-rank evidence but weaker as a broad candidate retriever.

## Recommendation

Do not fully replace fixed 500/50 yet. Use fixed 500/50 as the broad retrieval baseline, and keep section-aware v2 as the answer-evidence / reranking candidate. The next best architecture is two-stage: retrieve candidate documents with fixed 500/50, then use section-aware chunks from those documents to choose evidence and generate answers.

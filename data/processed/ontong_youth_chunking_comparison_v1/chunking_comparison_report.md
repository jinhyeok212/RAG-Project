# Ontong Youth Chunking Comparison v1

created_at: 2026-08-31T21:17:05+09:00

## Main Results

| chunking | chunks | retrieval | Hit@1 | Hit@5 | Hit@10 | Recall@5 | MRR | latency ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 300/50 | 1742 | Hybrid RRF | 0.6809 | 0.9362 | 0.9362 | 0.3856 | 0.7745 | 9.67 |
| 300/50 | 1742 | Chroma SVD | 0.6383 | 0.7872 | 0.8936 | 0.3516 | 0.7209 | 92.78 |
| 500/50 | 1060 | Hybrid RRF | 0.7021 | 0.9149 | 0.9574 | 0.5105 | 0.7812 | 4.75 |
| 500/50 | 1060 | Chroma SVD | 0.6596 | 0.8936 | 1.0000 | 0.5226 | 0.7525 | 94.50 |
| 700/100 | 797 | Hybrid RRF | 0.6809 | 0.9149 | 0.9362 | 0.6489 | 0.7773 | 6.13 |
| 700/100 | 797 | Chroma SVD | 0.6596 | 0.8936 | 0.9149 | 0.6601 | 0.7378 | 95.33 |

## Decision Scores

| rank | chunking | score | hybrid Hit@5 | hybrid MRR | Chroma Hit@5 | Chroma Hit@10 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 500/50 | 0.8889 | 0.9149 | 0.7812 | 0.8936 | 1.0000 |
| 2 | 700/100 | 0.8752 | 0.9149 | 0.7773 | 0.8936 | 0.9149 |
| 3 | 300/50 | 0.8521 | 0.9362 | 0.7745 | 0.7872 | 0.8936 |

## Recommendation

Use 500/50 as the balanced baseline for the next RAG iteration. It has the best combined score, the strongest Chroma Top-10 recall, and moderate index size. Keep 300/50 as a lexical-retrieval candidate and 700/100 as a high-recall/low-index-size candidate for later ablation.

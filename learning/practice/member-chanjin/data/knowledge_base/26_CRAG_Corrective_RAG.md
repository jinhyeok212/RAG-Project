# CRAG: Corrective Retrieval Augmented Generation

## 논문 정보

- 제목: Corrective Retrieval Augmented Generation
- 저자: Shi-Qi Yan 등
- 연도: 2024
- 링크: https://arxiv.org/abs/2401.15884

## 문제의식

RAG는 검색 문서가 좋아야 답변도 좋아진다. 하지만 검색이 잘못되면 LLM이 잘못된 문서를 바탕으로 그럴듯한 답변을 생성할 수 있다.

## 핵심 아이디어

CRAG는 검색된 문서 품질을 평가하는 retrieval evaluator를 둔다. 검색 결과가 충분히 좋으면 그대로 사용하고, 불확실하거나 낮은 품질이면 보완 검색 또는 다른 액션을 수행한다. 또한 문서를 분해하고 재조합해 불필요한 정보를 줄이는 접근을 사용한다.

## RAG와의 연결

CRAG는 검색 실패를 그냥 생성 단계로 넘기지 말고, 검색 단계에서 품질을 진단해야 한다는 점을 강조한다.

## 파라미터/실험 포인트

- retrieval confidence threshold.
- 검색 결과가 낮은 품질일 때 web search 또는 fallback retriever를 사용할지.
- 문서 filtering 전후 token 비용과 faithfulness 변화.

## Self-RAG와 차이

Self-RAG는 모델이 검색·생성·비판을 자기성찰 방식으로 수행하는 쪽에 가깝다. CRAG는 검색 결과 품질이 낮을 때 correction action을 수행하는 pipeline 관점이 더 강하다.

## 실무적 해석

우리 프로젝트에서는 `top1_similarity`, `avg_similarity_top3`, `retrieval_hit_rate`를 바탕으로 검색 결과가 위험한 질문을 표시하고, 실패 위험 예측 모델을 붙이면 CRAG 관점을 구현할 수 있다.

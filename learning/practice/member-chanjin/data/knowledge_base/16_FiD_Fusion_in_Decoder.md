# FiD: Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering

## 논문 정보

- 통칭: Fusion-in-Decoder, FiD
- 저자: Gautier Izacard, Edouard Grave
- 연도: 2020
- 링크: https://arxiv.org/abs/2007.01282

## 문제의식

RAG에서는 여러 passage를 검색하지만, 이 passage들을 생성 모델이 어떻게 결합할지가 중요하다. 단순히 모든 문서를 하나의 긴 context로 붙이면 문서 간 관계를 잘 활용하지 못하거나, 길이 제한 문제가 생긴다.

## 핵심 아이디어

FiD는 각 passage를 encoder에서 따로 처리하고, decoder 단계에서 여러 passage 정보를 결합한다. 즉, 문서들을 하나의 덩어리로 섞기보다, passage별 표현을 보존한 뒤 생성 단계에서 융합한다.

## RAG와의 연결

일반적인 앱 수준 RAG에서는 FiD 구조를 직접 구현하지는 않는다. 하지만 FiD는 `여러 검색 문서를 LLM에게 어떻게 제공할 것인가`라는 중요한 질문을 던진다.

## 파라미터/실험 포인트

- `top_k`를 늘리면 더 많은 passage를 줄 수 있지만, 모든 passage가 도움이 되는 것은 아니다.
- 검색 문서 순서와 문서 간 중복이 답변 품질에 영향을 준다.
- reranker를 쓰면 generator에게 줄 passage 품질을 개선할 수 있다.

## 실무적 해석

FiD는 “검색된 문서를 많이 넣는다고 무조건 좋은 것이 아니다”라는 점을 보여준다. RAG 대시보드에서는 top_k별 품질과 비용을 같이 봐야 한다.

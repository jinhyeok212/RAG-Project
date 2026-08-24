# DPR: Dense Passage Retrieval for Open-Domain Question Answering

## 논문 정보

- 제목: Dense Passage Retrieval for Open-Domain Question Answering
- 저자: Vladimir Karpukhin 등
- 연도: 2020
- 링크: https://arxiv.org/abs/2004.04906

## 문제의식

Open-domain QA에서는 질문에 답하기 위해 수많은 문서 중 관련 passage를 먼저 찾아야 한다. 과거에는 TF-IDF, BM25 같은 sparse retrieval이 표준처럼 쓰였다. 하지만 키워드가 정확히 일치하지 않으면 의미적으로 관련 있는 문서를 놓칠 수 있다.

## 핵심 아이디어

DPR은 질문과 passage를 각각 dense vector로 인코딩하는 dual-encoder 구조를 사용한다. 질문 벡터와 passage 벡터의 유사도를 계산해 관련 문서를 찾는다. 이를 통해 단순 키워드 일치보다 의미 기반 검색을 더 잘 수행할 수 있다.

## RAG와의 연결

현대 RAG에서 사용하는 embedding search의 핵심 배경이 DPR이다. 사용자가 질문을 입력하면 질문을 벡터로 만들고, 문서 chunk도 벡터로 만들어 cosine similarity나 dot product로 가까운 문서를 찾는다.

## 파라미터/실험 포인트

- `embedding_model`: 어떤 모델을 쓰느냐에 따라 retrieval 품질이 크게 달라진다.
- `top_k`: dense retrieval에서 정답 문서가 상위 몇 개 안에 들어오는지 평가해야 한다.
- `similarity_threshold`: 너무 높으면 필요한 문서를 놓치고, 너무 낮으면 노이즈 문서가 들어온다.
- `chunk_size`: passage 단위가 너무 작으면 문맥이 부족하고, 너무 크면 벡터가 여러 주제를 섞어 표현한다.

## 실무적 해석

DPR은 “RAG 성능의 상당 부분은 LLM보다 retriever가 결정한다”는 점을 보여준다. 그래서 LLMOps 대시보드에는 반드시 retrieval hit rate, MRR, context precision 같은 검색 품질 지표가 들어가야 한다.

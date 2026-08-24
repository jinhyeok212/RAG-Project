# 06. Retriever, Reranker, MMR

## Retriever란?

Retriever는 질문과 관련 있는 문서 chunk를 찾아오는 구성요소다. RAG에서 retriever가 실패하면 LLM은 잘못된 근거를 보고 답변하거나 근거 없이 추측하게 된다.

## Vector Retriever

Vector retriever는 질문과 문서 chunk를 임베딩한 뒤 벡터 유사도로 관련 문서를 찾는다. 의미 기반 검색에 유리하다.

## Keyword Retriever

Keyword retriever는 BM25나 TF-IDF처럼 단어 기반으로 관련 문서를 찾는다. 특정 용어, 제품명, 에러 코드, 정책명처럼 정확한 키워드가 중요한 경우 강점이 있다.

## Hybrid Search

Hybrid search는 vector search와 keyword search를 함께 쓰는 방식이다. 의미 검색과 키워드 검색의 장점을 결합한다. 실무에서는 제품명, 오류 코드, 정책명, 숫자 정보가 섞인 질문이 많아서 hybrid search가 유용한 경우가 많다.

## Reranker란?

Reranker는 1차 검색된 문서들을 질문과 다시 비교해 순서를 재정렬하는 모델이다. 예를 들어 vector search로 20개 문서를 가져온 뒤, reranker가 더 관련성 높은 5개를 골라 LLM에 전달할 수 있다.

Reranker를 쓰면 검색 품질이 좋아질 수 있지만, 추가 계산이 필요하므로 latency가 증가할 수 있다.

## MMR이란?

MMR은 Maximum Marginal Relevance의 약자다. 관련성만 높은 문서를 고르는 것이 아니라, 서로 너무 비슷한 문서가 반복되지 않도록 다양성도 고려한다.

질문에 대해 검색 결과 5개를 가져왔는데 모두 같은 문단의 반복이라면 LLM이 다양한 근거를 얻지 못한다. MMR은 중복 검색 결과를 줄이는 데 도움을 준다.

## mmr_lambda란?

mmr_lambda는 관련성과 다양성의 균형을 조절한다.

- 값이 1에 가까울수록 질문과의 관련성을 더 중요하게 본다.
- 값이 0에 가까울수록 검색 결과 간 다양성을 더 중요하게 본다.

실습에서는 0.5~0.8 사이를 추천한다. 문서가 짧고 중복이 적으면 MMR 효과가 작을 수 있다. 문서가 길고 유사한 chunk가 많으면 MMR이 유용하다.

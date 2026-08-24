# REPLUG: Retrieval-Augmented Black-Box Language Models

## 논문 정보

- 제목: REPLUG: Retrieval-Augmented Black-Box Language Models
- 저자: Weijia Shi 등
- 연도: 2023
- 링크: https://arxiv.org/abs/2301.12652

## 문제의식

많은 실무자는 LLM 내부 구조를 수정하거나 fine-tuning할 수 없다. API로만 접근 가능한 black-box LLM을 쓰는 경우가 많다. 그렇다면 검색 증강을 어떻게 적용할 수 있을까?

## 핵심 아이디어

REPLUG는 LLM을 black box로 두고, 검색된 문서를 입력 앞에 붙여 LLM 성능을 높이는 구조를 제안한다. 또한 LLM의 피드백을 이용해 retriever를 조정하는 관점도 포함한다.

## RAG와의 연결

대부분의 실무 RAG는 REPLUG와 비슷하다. OpenAI, Claude, Gemini 같은 API 모델의 내부를 바꾸지 않고, 외부 문서를 프롬프트에 넣어 답변을 생성한다.

## 파라미터/실험 포인트

- `top_k`와 context 길이는 API 비용에 직접 연결된다.
- LLM이 black box이면 retriever, reranker, prompt, logging이 더 중요해진다.
- 모델 내부를 바꾸지 못하므로 실패 분석은 로그 기반으로 해야 한다.

## 실무적 해석

우리 프로젝트는 black-box LLM 기반 RAG 운영에 가깝다. 따라서 답변 품질을 높이기 위해서는 모델 학습보다 retrieval 품질, prompt 구성, 평가 로그, 비용 모니터링이 핵심이다.

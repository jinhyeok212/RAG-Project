# RAG 운영 품질 진단 및 개선을 위한 LLMOps 플랫폼
## MVP 목적·사용자 정의·필수 기능 및 Baseline RAG 설계

---

## 1. 문서 목적

이 문서는 프로젝트의 공통 기준을 정하기 위한 PM 초안이다.

팀원 모두가 처음부터 각자 프로젝트 목적, 사용자, 기능 및 기술 구조를 조사한 뒤 합의하는 방식은 조사 기준이 서로 달라 의사결정이 어려워질 수 있다. 따라서 PM이 먼저 초안을 작성하고, 팀원들은 다음 기준으로 검토한다.

- 유지
- 수정
- 추가
- 삭제
- 보류

PM이 초안을 제시하지만 모든 내용을 혼자 확정하는 것은 아니다. 팀원들은 담당 분야의 기술적 가능성, 구현 난이도, 일정 및 필요성을 검토해 초안을 보완한다.

---

## 2. 프로젝트 배경

RAG(Retrieval-Augmented Generation)는 사용자의 질문과 관련된 문서를 먼저 검색한 뒤, 검색된 문서를 LLM에 제공하여 답변을 생성하는 구조이다.

일반적인 RAG 서비스에서는 최종 질문과 답변만 확인하는 경우가 많다. 그러나 답변 품질이 좋지 않을 때 최종 결과만으로는 다음 내용을 판단하기 어렵다.

- 질문과 관련된 문서가 검색되었는가?
- 정답 근거가 검색 결과의 상위에 배치되었는가?
- 검색된 문서에 필요한 근거가 포함되어 있었는가?
- 검색은 성공했지만 답변 생성 단계에서 문제가 발생했는가?
- LLM이 검색 문서에 없는 내용을 생성했는가?
- 응답시간이나 Token 사용량이 지나치게 증가했는가?
- 설정을 변경한 뒤 실제 품질이 개선되었는가?

따라서 RAG 서비스의 품질을 개선하려면 최종 답변뿐만 아니라 검색 과정, 생성 과정, 설정값, 평가 결과를 함께 추적해야 한다.

---

## 3. 프로젝트 문제 정의

> RAG 서비스에서 최종 질문과 답변만 확인해서는 품질 저하의 원인이 검색 단계인지, 답변 생성 단계인지 판단하기 어렵다. 본 프로젝트는 RAG 실행 과정의 Trace와 품질 평가 결과를 수집하고, 실패 원인과 위험 수준 및 개선 방향을 제공하는 LLMOps 관리자 플랫폼을 구축하는 것을 목표로 한다.

---

## 4. 프로젝트 목적

### 4.1 핵심 목적

본 프로젝트의 핵심 목적은 다음과 같다.

1. RAG 실행 과정에서 발생하는 정보를 수집한다.
2. 검색 품질과 답변 품질을 분리하여 평가한다.
3. 평가 결과를 바탕으로 실패 원인을 진단한다.
4. 운영자가 이해하기 쉬운 위험 수준과 개선 방향을 제공한다.
5. 설정 변경 전후의 품질, 비용 및 응답시간을 비교할 수 있게 한다.

### 4.2 구현 관점의 목표

```text
RAG 실행
→ Trace 수집
→ 검색·답변 평가
→ 실패 원인 판단
→ 대시보드 확인
→ 설정 변경
→ 변경 전후 비교
```

### 4.3 학습 관점의 목표

프로젝트 결과물뿐 아니라 팀원들이 다음 내용을 이해하는 것을 목표로 한다.

- RAG 전체 파이프라인
- 데이터 전처리와 Chunking
- Embedding과 Vector DB
- Retriever와 Generator의 역할
- 검색 실패와 생성 실패의 차이
- RAG Trace와 운영 로그
- 검색 및 답변 평가 지표
- 품질·비용·응답시간의 관계
- Decision Engine의 규칙 설계
- 실험 조건 통제와 결과 비교

---

## 5. 사용자 정의

### 5.1 주요 사용자

RAG 기반 서비스를 개발하거나 운영하는 개발자 및 관리자

### 5.2 사용자가 겪는 문제

RAG 답변이 좋지 않을 때 운영자는 다음 문제를 겪는다.

- 잘못된 문서가 검색되었는지 알기 어렵다.
- 정답 문서가 검색되었지만 순위가 낮았는지 알기 어렵다.
- 검색 문서에 근거가 있었는데 LLM이 잘못 답했는지 판단하기 어렵다.
- Prompt, Top-k, Chunk Size 등의 설정 중 무엇을 수정해야 하는지 알기 어렵다.
- 변경한 설정이 실제 품질 개선으로 이어졌는지 비교하기 어렵다.
- 품질 개선 과정에서 응답시간이나 Token 사용량이 얼마나 증가했는지 파악하기 어렵다.

### 5.3 주요 사용자 시나리오

```text
1. 운영자가 RAG 질문을 실행한다.
2. 질문, 검색 문서, 검색 점수, 생성 답변 및 설정값이 저장된다.
3. 검색 품질과 답변 품질이 평가된다.
4. Decision Engine이 위험 수준과 실패 유형을 판단한다.
5. 운영자가 대시보드에서 실행 결과와 판단 근거를 확인한다.
6. 추천 내용을 참고해 Top-k, Chunk Size, Prompt 등의 설정을 변경한다.
7. 변경 전후의 품질, 응답시간 및 Token 사용량을 비교한다.
```

---

## 6. MVP 정의

### 6.1 MVP 목표

MVP에서는 모든 LLMOps 기능을 구현하지 않는다.

하나의 공통 Baseline RAG를 대상으로 다음 흐름이 하나의 시스템에서 연결되는 것을 MVP의 핵심으로 정의한다.

```text
질문 실행
→ 관련 문서 검색
→ 답변 생성
→ Trace 저장
→ 품질 평가
→ 실패 원인 판단
→ 관리자 화면 조회
```

### 6.2 핵심 MVP 완료 시점

5주 차까지 위 흐름이 연결된 상태를 핵심 MVP 완료로 본다.

6~8주 차에는 설정 실험, 기능 확장, 통합 테스트, 배포 및 발표 준비를 진행한다.

---

## 7. MVP 필수 기능

### 7.1 공통 Baseline RAG

팀에서 공통으로 사용할 RAG 파이프라인을 구현한다.

필수 기능은 다음과 같다.

- 데이터 불러오기
- 공통 데이터 형식 변환
- 문서 전처리
- Chunking
- 문서 Embedding
- Vector DB 저장 및 불러오기
- 질문 Embedding
- 유사 문서 검색
- Top-k 검색 결과 반환
- 검색 문서를 Context로 구성
- LLM 답변 생성
- 전체 RAG 파이프라인 실행

### 7.2 Trace 및 로그 저장

질문 실행마다 고유한 Trace를 생성하고 다음 정보를 저장한다.

#### 요청 정보

- Trace ID
- 질문
- 실행 시각
- 데이터셋 또는 데이터 버전

#### 검색 정보

- 검색된 문서 ID
- 검색된 Chunk ID
- 검색 순위
- Similarity Score
- 검색된 Chunk 내용
- 검색 응답시간

#### 생성 정보

- 생성 답변
- 사용한 LLM
- Prompt Version
- 입력 Token
- 출력 Token
- 생성 응답시간

#### 설정 정보

- Embedding 모델
- Vector DB
- 검색 방식
- Top-k
- Chunk Size
- Chunk Overlap

#### 운영 정보

- 전체 응답시간
- 오류 여부
- 오류 메시지

### 7.3 검색 품질 평가

초기 MVP에서는 다음 지표를 우선 구현한다.

- Hit@k
- MRR

진행 상황에 따라 다음 지표를 추가한다.

- Context Precision
- Context Recall

### 7.4 답변 품질 평가

초기 MVP에서는 다음 평가 방법을 검토한다.

- Faithfulness
- Answer Relevancy
- 정답과 생성 답변의 의미 유사도

답변 평가 도구, API 비용 및 구현 난이도를 확인한 뒤 적용 범위를 확정한다.

### 7.5 규칙 기반 Decision Engine

여러 평가 결과를 조합해 운영자가 이해하기 쉬운 형태로 변환한다.

출력 항목은 다음과 같다.

- Quality Score
- Risk Level
- Failure Type
- 판단 근거
- Recommended Action

초기 실패 유형은 다음과 같이 구성한다.

- Retrieval Failure
- Ranking Failure
- Evidence Missing
- Unsupported Answer Risk
- Relevancy Failure
- Performance Issue
- Cost Inefficiency

규칙 예시는 다음과 같다.

```text
Hit@k가 낮음
→ Retrieval Failure
→ Embedding, Chunking 또는 검색 방식 점검

Hit@k는 성공했지만 MRR이 낮음
→ Ranking Failure
→ 검색 순위 개선 또는 Reranker 적용 검토

검색은 성공했지만 Faithfulness가 낮음
→ Unsupported Answer Risk
→ Prompt 또는 생성 모델 점검

응답시간이 기준보다 높음
→ Performance Issue
→ Top-k, 모델 또는 검색 구조 점검

Token 사용량이 기준보다 높음
→ Cost Inefficiency
→ Context 길이와 Prompt 구조 점검
```

### 7.6 관리자 대시보드

최소한 다음 내용을 확인할 수 있어야 한다.

#### 전체 현황

- 전체 실행 건수
- 성공 및 오류 건수
- 평균 전체 응답시간
- 평균 검색 응답시간
- 평균 생성 응답시간
- 평균 Token 사용량
- 평균 검색 품질
- 평균 답변 품질

#### 품질 및 실패 현황

- Risk Level 분포
- Failure Type 분포
- 검색 품질 지표
- 답변 품질 지표
- 낮은 품질의 실행 목록

#### Trace 상세

- 질문
- 검색된 문서
- 검색 점수와 순위
- 생성 답변
- 평가 결과
- 실패 유형
- 판단 근거
- 추천 조치
- 사용 설정

### 7.7 실험 비교

Baseline과 변경된 설정의 결과를 비교한다.

비교 항목은 다음과 같다.

- 검색 품질
- 답변 품질
- 응답시간
- Token 사용량
- 실패 유형 변화

---

## 8. 확장 기능

핵심 MVP 완성 후 다음 기능 중 일부를 선택한다.

- Top-k 비교
- Chunk Size 비교
- Chunk Overlap 비교
- Embedding 모델 비교
- Prompt Version 비교
- Dense Search와 Hybrid Search 비교
- Reranker 적용
- 답변 거절 Prompt
- 질문 유형별 실패 분석
- 문서 유형별 실패 분석
- 사용자 피드백
- AI Hub 데이터 적용
- 간단한 이상 요청 탐지

모든 기능을 구현하지 않고 프로젝트 진행 상황에 따라 2~3개의 실험과 1개의 선택 확장 기능을 우선한다.

---

## 9. MVP 제외 기능

2개월 동안 핵심 구조와 확장 실험까지 완료하기 위해 다음 기능은 초기 범위에서 제외한다.

- Agent
- Multi-Agent
- Fine-tuning
- Kubernetes
- 자동 재학습
- 여러 외부 RAG 서비스의 실시간 연결
- 완전 자동 설정 최적화
- 복잡한 실시간 이상 탐지
- 대규모 PyTorch 예측 모델

PyTorch 모델은 검수된 라벨과 충분한 데이터가 확보되었을 때만 선택적으로 검토한다.

---

## 10. Baseline RAG 설계 원칙

Baseline RAG는 가장 높은 성능을 내는 시스템이 아니라 이후 실험의 기준이 되는 시스템이다.

따라서 다음 원칙을 적용한다.

1. 구조가 단순해야 한다.
2. 모든 팀원이 같은 방식으로 실행할 수 있어야 한다.
3. 설정값이 명확하게 기록되어야 한다.
4. 동일한 조건으로 재실행할 수 있어야 한다.
5. 기능을 모듈 단위로 변경할 수 있어야 한다.
6. Trace와 평가 기능을 연결하기 쉬워야 한다.

---

## 11. Baseline RAG 전체 흐름

### 11.1 인덱스 구축 과정

```text
원본 데이터
→ 데이터 로딩
→ 공통 데이터 형식 변환
→ 전처리
→ Chunking
→ Chunk Embedding
→ Vector DB 저장
→ Metadata 저장
```

### 11.2 질문 처리 과정

```text
사용자 질문
→ 질문 Embedding
→ Vector DB 유사도 검색
→ Top-k Chunk 반환
→ Context 구성
→ Prompt 구성
→ LLM 답변 생성
→ Trace 저장
→ 평가 실행
→ Decision Engine 판단
```

---

## 12. 초기 데이터 활용 초안

### 12.1 사전 학습 단계

- AI Hub 데이터 사용: 3명
- KorQuAD 데이터 사용: 3명

이 단계는 데이터셋별 성능 비교가 아니라 RAG 전체 흐름을 직접 구현해보는 학습 단계로 본다.

### 12.2 공통 Baseline 데이터

초기 공통 Baseline은 KorQuAD 일부 데이터를 우선 활용하는 방안을 제안한다.

선정 이유는 다음과 같다.

- 질문, 정답 및 문서의 관계를 확인하기 쉽다.
- 정답 근거를 이용한 검색 평가 구조를 만들기 상대적으로 수월하다.
- 개인 구현 경험을 공통 구조로 전환하기 쉽다.

AI Hub 데이터는 공통 파이프라인이 안정된 뒤 다음 목적으로 활용할 수 있다.

- 다른 데이터 구조 적용
- 정답 없음 질문 실험
- 실제 도메인 적용 가능성 확인
- 데이터셋별 일반화 확인

최종 데이터 선정은 팀 회의에서 확정한다.

---

## 13. 초기 기술 설정 초안

| 항목 | 초기안 |
|---|---|
| 개발 언어 | Python |
| 초기 데이터 | KorQuAD 일부 |
| 검색 방식 | Dense Search |
| Vector DB | FAISS 또는 Chroma |
| Top-k | 3 |
| Chunk Size | 공통 기준 1개 |
| Chunk Overlap | Chunk Size의 약 10% |
| Embedding 모델 | 한국어 지원 모델 1개 |
| Generator | 팀 공통 사용이 가능한 LLM 1개 |
| Prompt | 제공된 Context에 근거해 답하도록 구성 |
| 실행 방식 | 콘솔 또는 API |
| 설정 관리 | YAML, JSON 또는 Python Config |
| 로그 저장 | 초기 파일 저장 후 DB 확장 가능 |
| 평가 실행 | Trace 저장 후 평가 모듈 호출 |

초기값이 최적이라는 의미는 아니다. 6~7주 차 실험을 위한 기준값이다.

---

## 14. 권장 디렉터리 구조

```text
project/
├─ data/
│  ├─ raw/
│  └─ processed/
│
├─ src/
│  ├─ data_loader.py
│  ├─ preprocessing.py
│  ├─ chunking.py
│  ├─ embedding.py
│  ├─ vector_store.py
│  ├─ retriever.py
│  ├─ generator.py
│  ├─ rag_pipeline.py
│  ├─ trace_logger.py
│  └─ config.py
│
├─ evaluation/
│  ├─ retrieval_metrics.py
│  ├─ answer_metrics.py
│  └─ evaluator.py
│
├─ decision/
│  └─ decision_engine.py
│
├─ backend/
├─ dashboard/
├─ experiments/
├─ configs/
├─ tests/
├─ docs/
└─ README.md
```

---

## 15. 주요 모듈 역할

| 모듈 | 역할 |
|---|---|
| `data_loader.py` | 원본 데이터 불러오기 |
| `preprocessing.py` | 데이터를 공통 형식으로 변환 |
| `chunking.py` | 문서를 Chunk 단위로 분할 |
| `embedding.py` | 문서와 질문을 Vector로 변환 |
| `vector_store.py` | Vector DB 생성·저장·불러오기 |
| `retriever.py` | 질문과 유사한 Chunk 검색 |
| `generator.py` | 검색된 Context를 바탕으로 답변 생성 |
| `rag_pipeline.py` | RAG 전체 실행 순서 관리 |
| `trace_logger.py` | 질문, 검색, 생성 및 설정 정보 저장 |
| `retrieval_metrics.py` | Hit@k, MRR 등 검색 평가 |
| `answer_metrics.py` | Faithfulness 등 답변 평가 |
| `evaluator.py` | 검색·답변 평가 실행 및 결과 통합 |
| `decision_engine.py` | 평가 결과 기반 실패 유형과 조치 판단 |

---

## 16. 공통 데이터 스키마 초안

```json
{
  "document_id": "doc_001",
  "title": "문서 제목",
  "content": "원본 문서 내용",
  "source": "korquad",
  "metadata": {}
}
```

질문 데이터 예시는 다음과 같다.

```json
{
  "question_id": "q_001",
  "question": "질문 내용",
  "ground_truth_answer": "정답",
  "ground_truth_document_id": "doc_001",
  "ground_truth_context": "정답 근거 문장"
}
```

Chunk 데이터 예시는 다음과 같다.

```json
{
  "chunk_id": "doc_001_chunk_001",
  "document_id": "doc_001",
  "chunk_text": "분할된 문서 내용",
  "chunk_index": 0,
  "metadata": {}
}
```

---

## 17. Trace 스키마 초안

```json
{
  "trace_id": "trace_001",
  "question_id": "q_001",
  "question": "질문 내용",
  "retrieved_chunks": [
    {
      "rank": 1,
      "chunk_id": "doc_001_chunk_001",
      "document_id": "doc_001",
      "score": 0.85,
      "text": "검색된 문서 내용"
    }
  ],
  "generated_answer": "생성된 답변",
  "settings": {
    "embedding_model": "model_name",
    "vector_db": "faiss",
    "top_k": 3,
    "chunk_size": 500,
    "chunk_overlap": 50,
    "prompt_version": "v1",
    "llm_model": "model_name"
  },
  "performance": {
    "retrieval_latency_ms": 50,
    "generation_latency_ms": 1200,
    "total_latency_ms": 1250,
    "input_tokens": 500,
    "output_tokens": 100
  },
  "error": null
}
```

---

## 18. 평가 및 진단 결과 스키마 초안

```json
{
  "trace_id": "trace_001",
  "retrieval_metrics": {
    "hit_at_k": 1,
    "mrr": 1.0
  },
  "answer_metrics": {
    "faithfulness": 0.9,
    "answer_relevancy": 0.85,
    "semantic_similarity": 0.88
  },
  "decision": {
    "quality_score": 87,
    "risk_level": "LOW",
    "failure_type": "NONE",
    "reason": "검색 및 답변 품질이 기준 이상입니다.",
    "recommended_action": "현재 설정을 유지합니다."
  }
}
```

---

## 19. 팀 검토 항목

팀원은 본 문서를 읽고 다음 의견을 준비한다.

1. 반드시 추가해야 하는 기능
2. 삭제하거나 후순위로 두어야 하는 기능
3. 기술적으로 구현하기 어려운 부분
4. 설명이 부족하거나 이해되지 않는 부분
5. 본인이 맡을 수 있는 기능
6. Baseline 설정에서 변경이 필요한 부분
7. 평가 지표의 현실적인 구현 가능성
8. 5주 차까지 구현 가능한 범위인지 여부

---

## 20. 최종 완료 기준

핵심 MVP는 다음 시나리오가 동작할 때 완료된 것으로 본다.

```text
1. 사용자가 질문을 실행한다.
2. RAG가 관련 문서를 검색한다.
3. 검색 문서를 기반으로 답변을 생성한다.
4. 실행 Trace가 저장된다.
5. 검색 및 답변 품질이 평가된다.
6. Decision Engine이 실패 유형과 위험 수준을 판단한다.
7. 관리자가 대시보드에서 실행 결과를 확인한다.
```

이후 확장 단계에서 설정을 변경하고 Baseline과 결과를 비교한다.

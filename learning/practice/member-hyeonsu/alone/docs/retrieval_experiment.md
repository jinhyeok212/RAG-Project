# 슈퍼 카테고리 Retrieval-only 검색 실험

## 실험 목적

이 실험은 LLM이 답변을 생성하는 시스템이 아니라, 사용자 질문과 의미가 가까운 기존 고객 질문을 찾는 **검색 전용 baseline**이다. 기본 모델 `jhgan/ko-sroberta-multitask`는 한국어 문장 의미 유사도 검색 가능성을 처음 확인하기 위한 기준 모델이며, 가장 좋은 모델이라고 단정하지 않는다. `.env`의 `EMBEDDING_MODEL`을 바꾸어 이후 모델을 비교할 수 있다.

## 질문을 임베딩한다는 의미

임베딩은 문장을 고정 길이 숫자 벡터로 바꾸는 과정이다. 표현이 달라도 의미가 비슷한 질문이 벡터 공간에서 가까워지도록 학습된 SentenceTransformer를 사용한다. 모델이 지원하는 `normalize_embeddings=True`로 벡터를 L2 정규화한다.

첫 baseline에서는 Training의 `question`만 임베딩한다. 답변을 질문에 합치면 답변 표현이 검색 벡터에 영향을 줄 수 있으므로, 먼저 순수한 질문 검색 성능을 측정하고 `question+answer` 방식은 다음 비교 실험으로 남긴다.

## Chroma의 역할과 cosine 거리

Chroma는 Training 질문 벡터, 원문 질문, 답변과 메타데이터를 저장하고 질의 벡터와 가까운 문서를 빠르게 찾는 Vector DB다. 이 실험의 컬렉션은 `hnsw:space=cosine`으로 만든다. 문장 벡터의 방향이 얼마나 가까운지 비교하기에 적합하고 정규화된 문장 임베딩과 함께 쓰기 쉽다.

중요하게도 Chroma가 반환하는 값은 **유사도가 아니라 거리**다.

- 거리는 낮을수록 벡터가 가깝다.
- 거리와 유사도를 혼동하면 안 된다.
- 이 실험은 임의의 공식으로 부정확한 “유사도 점수”를 만들지 않고 원래 `distance`를 그대로 기록한다.

## Training과 Validation의 역할

- Training 5,000개 질문은 Chroma에 저장되는 검색 문서다.
- Validation 500개 질문은 Chroma에 넣지 않고 검색 질의로만 사용한다.
- Validation의 원본 답변은 검색 결과를 사람이 확인하기 위한 참고값이다.

두 분할은 동일한 `document_id`를 공유하지 않는다. 따라서 현재는 특정 Training 문서 ID를 정답으로 간주하지 않고 인텐트 일치를 대리 지표로 사용한다.

## 평가 지표

**Intent Hit@k**는 상위 k개 검색 결과 중 Validation 질문의 `expected_intent`와 같은 Training 인텐트가 하나라도 있는 질문의 비율이다. Hit@1, Hit@3, Hit@5를 계산한다.

**Intent MRR**은 같은 인텐트가 처음 등장한 순위의 역수다. 첫 순위면 1, 두 번째면 1/2이며 평가 깊이 안에 없으면 0이다. 실패 분석을 위해 최대 20위까지 찾되 공식 Hit 지표는 1, 3, 5에서만 계산한다.

인텐트 평가는 정답 문서 연결 라벨 없이도 주제 수준 검색을 빠르게 확인할 수 있다는 장점이 있다. 그러나 다음 한계가 있다.

- 같은 인텐트라도 상품, 가격, 배송 조건이 다르면 실제 답변은 틀릴 수 있다.
- 같은 질문에 서로 다른 답변이 존재하면 질문 벡터만으로 어떤 답변이 맞는지 결정하기 어렵다.
- 인텐트가 일치해도 실제로 유용하거나 사실에 맞는 답변이라는 뜻은 아니다.
- 질문 수가 적은 인텐트의 지표는 표본 한두 개에 크게 변하므로 해석에 주의해야 한다.
- 실패 유형 CSV의 분류는 규칙 기반 후보이며 사람이 최종 판단해야 한다.

## 환경 설정

`.env` 기본값은 다음과 같다.

```env
EMBEDDING_MODEL=jhgan/ko-sroberta-multitask
CHROMA_PATH=./chroma_db
CHROMA_COLLECTION=super_questions
TOP_K=5
EVALUATION_DEPTH=20
EMBEDDING_BATCH_SIZE=64
```

## Windows PowerShell 실행 순서

프로젝트 루트에서 실행한다. 시스템에 따라 `python` 대신 실제 Python 경로를 사용할 수 있다.

```powershell
python -m pip install sentence-transformers chromadb pandas numpy python-dotenv
python scripts/build_chroma_index.py
python scripts/search_demo.py
python scripts/evaluate_retrieval.py
```

1. 설치 명령은 필요한 라이브러리를 준비한다.
2. 인덱스 생성은 모델명, 임베딩 차원, 5,000개 문서의 시간과 Chroma 저장 검증 결과를 출력하고 `chroma_db/`를 만든다. 같은 `document_id`는 `upsert`되어 중복 추가되지 않는다.
3. 검색 데모는 한국어 질문을 받아 기본 상위 5개의 순위, 거리, 기존 질문, 답변, 인텐트와 카테고리를 출력한다.
4. 평가는 실제 측정 지표를 콘솔에 출력하고 `results/retrieval/`에 상세·요약·전체·인텐트별 지표와 실패 후보 CSV를 만든다.

## 결과 파일

- `super_validation_retrieval_results.csv`: 질문별 검색 결과와 원래 cosine 거리
- `super_validation_query_summary.csv`: 질문별 Hit와 MRR, 시간
- `super_retrieval_metrics.csv`: 전체 실제 측정 지표
- `super_intent_metrics.csv`: 인텐트별 실제 측정 지표
- `super_retrieval_failure_candidates.csv`: 사람이 검토할 실패 원인 후보

## 실제 baseline 실행 결과

아래 값은 이 프로젝트 환경에서 코드를 실제 실행해 측정한 결과다. 다른 장비나 실행 시점에는 시간이 달라질 수 있다.

### Chroma 인덱스 생성

| 항목 | 실제 값 |
|---|---:|
| 모델 | `jhgan/ko-sroberta-multitask` |
| 임베딩 차원 | 768 |
| 임베딩 문서 수 | 5,000 |
| 전체 문서 임베딩 시간 | 319.224초 |
| 평균 문서 임베딩 시간 | 63.845ms |
| Chroma 저장 문서 수 | 5,000 |

### Validation 검색 평가

| 지표 | 실제 값 |
|---|---:|
| Validation 질문 수 | 500 |
| Intent Hit@1 | 0.452000 |
| Intent Hit@3 | 0.604000 |
| Intent Hit@5 | 0.676000 |
| Intent MRR | 0.551851 |
| 평균 질문 임베딩 시간 | 56.089ms |
| 평균 검색 시간 | 11.233ms |
| 전체 평가 시간 | 40.315초 |

전체 평가 시간은 평가 함수 안에서 Chroma와 모델을 준비하고, 500개 질문을 임베딩·검색·집계하는 시간을 포함한다. Python 프로세스가 시작되어 라이브러리를 import하는 시간은 포함하지 않는다. MRR과 실패 분석의 첫 일치 순위는 최대 20위까지 확인했으며 Hit 지표는 정의대로 1·3·5위에서 계산했다.

## 다음 실험 계획

1. `question` 임베딩과 `question+answer` 임베딩 비교
2. 검색 결과 Top-k 3과 5 비교
3. 짧은 질문 포함 여부 비교
4. 다른 한국어 임베딩 모델 비교
5. 사람이 직접 관련 Training 문서 ID를 지정한 평가 데이터 구축

LLM, OpenAI·Gemini·Ollama 같은 생성 모델, 외부 API와 LangChain은 이 실험에서 사용하지 않는다.

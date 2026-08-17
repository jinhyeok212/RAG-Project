# 과거 상담 사례 검색 기반 RAG 구현 상세 보고서

작성 위치: `C:\말똥가리`
대상 데이터: `소상공인 고객 주문 질의-응답 텍스트`
현재 구현 단계: 카페 카테고리 MVP 검색 RAG

## 1. 전체 구현 범위

이번 구현은 사용자가 고객 질문을 입력하면 과거 상담 데이터에서 유사한 고객 질문과 상담사 답변을 검색하고, 검색된 과거 상담 답변만을 근거로 응답을 구성하는 RAG MVP이다.

현재 구현된 범위는 다음과 같다.

- 데이터 구조 분석
- 상담 발화에서 질문-답변 QA 문서 복원
- 개인정보 추정 패턴 마스킹
- Train 검색 corpus 생성
- Validation 평가 query 생성
- BM25 검색
- Dense Exact 검색
- Dense ANN 검색
- Hybrid RRF 검색
- Reranker fallback 검색
- 검색 성능 평가
- FastAPI API 뼈대
- Streamlit UI 뼈대
- 로컬 템플릿 기반 답변 생성

아직 구현하지 않은 범위는 다음과 같다.

- `BAAI/bge-m3` 실제 임베딩 모델
- `BAAI/bge-reranker-v2-m3` 실제 Cross-Encoder reranker
- OpenAI 또는 로컬 LLM 기반 자연어 생성
- RAGAS 기반 답변 품질 평가
- 전체 카테고리 전체 데이터 확장

## 2. RAG 파이프라인 구조

현재 RAG 흐름은 아래와 같다.

```text
원천 CSV
→ 데이터 구조 분석
→ 상담번호/상담내순번 기준 대화 정렬
→ QA번호/QA여부 기준 질문-답변 연결
→ 이전 문맥 1~3개 발화 연결
→ 개인정보 마스킹
→ retrieval_text 생성
→ Train corpus 저장
→ BM25 / Dense / ANN 인덱스 생성
→ 사용자 질문 입력
→ 검색 방식 선택
→ 과거 상담 QA 검색
→ 상위 상담 답변을 근거로 응답 생성
→ 검색 결과와 점수 반환
```

## 3. 데이터 구조 분석

구현 파일:

- `src/data/profile_data.py`

입력 데이터는 이미 압축 해제된 CSV 형태로 존재했다.

- `소상공인 고객 주문 질의-응답 텍스트/Training/*.csv`
- `소상공인 고객 주문 질의-응답 텍스트/Validation/*.csv`

확인된 핵심 컬럼은 다음과 같다.

| 컬럼 | 역할 |
| --- | --- |
| `IDX` | 파일 내부 행 번호 |
| `발화자` | 고객/상담사 구분 |
| `발화문` | 실제 발화 텍스트 |
| `카테고리` | 업종/도메인 |
| `QA번호` | 질문-답변 연결 단위 |
| `QA여부` | 질문(q) 또는 답변(a) 구분 |
| `감성` | 감성 라벨 |
| `인텐트` | 상담 의도 |
| `상담번호` | 하나의 상담 세션 ID |
| `상담내순번` | 상담 안에서 발화 순서 |

데이터 분석 결과는 아래 파일로 저장했다.

- `reports/data_profile.md`
- `reports/data_profile.json`

전체 파일이 크기 때문에 프로파일링은 파일당 최대 100,000행 샘플 기준으로 수행했다. 이 제한은 `configs/mvp.yaml`의 `profile_max_rows_per_file`에 기록되어 있다.

## 4. QA 문서 재구성

구현 파일:

- `src/data/build_corpus.py`

CSV 한 행을 바로 검색 문서로 쓰지 않았다. 실제 상담 흐름을 복원한 뒤 질문과 답변을 연결했다.

처리 방식은 다음과 같다.

1. `상담번호`로 상담 세션을 그룹화한다.
2. 같은 상담 안에서 `상담내순번` 기준으로 발화를 정렬한다.
3. `QA여부 = q` 또는 `발화자 = c`이면 고객 질문 후보로 본다.
4. `QA여부 = a` 또는 `발화자 = s`이면 상담사 답변 후보로 본다.
5. 같은 `QA번호` 안에서 앞선 질문과 뒤따르는 답변을 연결한다.
6. 질문만으로 의미가 부족할 수 있으므로 이전 1~3개 발화를 `previous_context_original`에 저장한다.
7. 질문이나 답변이 없는 경우 `quality_flags`에 결측 flag를 남긴다.
8. 원문 필드는 보존하고, 검색용 normalized 필드는 별도로 만든다.

생성된 QA 문서 구조는 다음과 같다.

| 필드 | 설명 |
| --- | --- |
| `doc_id` | 검색 문서 고유 ID |
| `source_split` | train 또는 validation |
| `category` | 카테고리 |
| `intent` | 인텐트 |
| `consultation_id` | 상담번호 |
| `qa_id` | QA번호 |
| `question_original` | 마스킹된 원본 고객 질문 |
| `answer_original` | 마스킹된 원본 상담사 답변 |
| `previous_context_original` | 이전 문맥 |
| `question_normalized` | 검색/분석용 질문 |
| `answer_normalized` | 검색/분석용 답변 |
| `retrieval_text` | 검색 인덱스에 들어가는 문서 텍스트 |
| `entities` | 가격, 수량, 장소, 상품명 등 엔티티 |
| `quality_flags` | 결측/개인정보 등 품질 flag |
| `metadata` | 발화 순서, 발화자 등 부가 정보 |

## 5. 개인정보 마스킹

구현 파일:

- `src/data/privacy_masking.py`

검색 인덱스에 넣기 전에 다음 패턴을 정규식으로 마스킹했다.

| 패턴 | 치환값 |
| --- | --- |
| 이메일 | `[EMAIL]` |
| 휴대전화 | `[PHONE]` |
| 유선전화 | `[TEL]` |
| 주민등록번호 형태 | `[RRN]` |
| 주문/예약/송장 번호 추정값 | `[ORDER_ID]` |

주의할 점은 데이터 원문 자체를 의미적으로 고치지 않았다는 것이다. 검색과 저장 전에 개인정보로 추정되는 문자열만 마스킹했다.

## 6. Retrieval Text 구성

검색 문서로 쓰는 `retrieval_text`는 단순히 발화문 하나만 넣지 않고, 검색 성능 비교가 가능하도록 버전 구조를 만들었다.

요청에서 제시한 4가지 버전은 아래와 같다.

| 버전 | 구성 |
| --- | --- |
| A | 질문만 |
| B | 이전 문맥 + 질문 |
| C | 카테고리 + 인텐트 + 이전 문맥 + 질문 |
| D | 카테고리 + 인텐트 + 이전 문맥 + 질문 + 답변 |

현재 MVP 기본값은 `C`이다.

설정 위치:

- `configs/mvp.yaml`
- `data.retrieval_text_version: "C"`

선택 이유는 검색 단계에서 답변 내용을 직접 넣으면 유사한 답변 문장 때문에 검색이 쉬워질 수 있지만, 실제 사용자 질문 기반 검색 성능을 과대평가할 수 있기 때문이다. 따라서 MVP에서는 카테고리, 인텐트, 이전 문맥, 고객 질문만 검색 문서로 사용했다.

## 7. MVP 데이터 범위

현재 전체 데이터를 처음부터 모두 인덱싱하지 않고, 하나의 카테고리에서 MVP를 검증했다.

설정:

| 항목 | 값 |
| --- | --- |
| MVP 카테고리 | `카페` |
| Train QA 문서 수 | 20,000 |
| Validation query 수 | 2,000 |
| 이전 문맥 발화 수 | 3 |
| 검색 텍스트 버전 | C |

생성 파일:

- `data/processed/train_rag_corpus.parquet`
- `data/processed/validation_rag_corpus.parquet`
- `data/processed/train_rag_corpus_sample.jsonl`
- `data/processed/validation_retrieval_queries.jsonl`
- `reports/data_quality_report.md`

## 8. BM25 Sparse Retrieval

구현 파일:

- `src/retrieval/bm25_retriever.py`
- `src/retrieval/tokenization.py`

BM25는 키워드 기반 sparse retrieval이다. 고객 주문/상담 데이터에는 상품명, 수량, 결제, 포장, 배송 같은 명확한 단어가 중요하므로 기본 검색기로 구현했다.

초기에는 모든 문서를 매 쿼리마다 훑는 단순 방식이었지만, 평가 시간이 너무 길어져 inverted index 방식으로 최적화했다.

현재 BM25 구현 특징:

- 한국어 형태소 분석기 없이 동작
- 공백 토큰 + 2~3글자 char n-gram을 섞은 `mixed` tokenizer 사용
- `k1 = 1.5`, `b = 0.75`
- 토큰별 posting list를 만들어 쿼리 토큰이 포함된 문서만 점수 계산
- 결과에 `bm25_rank`, `bm25_score` 반환

## 9. Dense Exact Retrieval

구현 파일:

- `src/retrieval/dense_retriever.py`

요청에서는 `BAAI/bge-m3` 같은 bi-encoder 임베딩 모델을 기본 후보로 두었지만, 현재 실행 환경에는 처음에 `sentence-transformers`가 없었다. 그래서 MVP가 바로 동작하도록 `sklearn` 기반 fallback을 구현했다.

현재 Dense Exact 구현:

- `TfidfVectorizer`
- `analyzer = char_wb`
- char n-gram 2~4
- max features 80,000
- L2 normalize
- cosine similarity에 해당하는 내적 계산
- 정확 검색 방식으로 전체 문서와 query vector 유사도 계산

즉 지금 Dense는 진짜 딥러닝 임베딩은 아니고, bi-encoder 형태를 흉내 내는 lightweight fallback이다. 나중에 `BAAI/bge-m3`로 교체할 수 있도록 downstream schema는 동일하게 맞춰두었다.

## 10. Dense ANN Retrieval

구현 파일:

- `src/retrieval/ann_index.py`

ANN은 정답 모델이 아니라 검색을 빠르게 하기 위한 근사 검색 방식이다.

현재 ANN 구현:

- Dense Exact TF-IDF matrix를 입력으로 사용
- `TruncatedSVD`로 128차원 축소
- `NearestNeighbors(metric="cosine")`로 근사형 인덱스 구성
- 결과에 `dense_rank`, `dense_score` 반환

현재는 FAISS/HNSW가 아니라 sklearn 기반 ANN fallback이다. Exact 검색 결과가 정상인지 확인한 뒤 FAISS HNSW 또는 IVF로 확장할 수 있게 분리해두었다.

## 11. Hybrid RRF Retrieval

구현 파일:

- `src/retrieval/rrf.py`
- `src/retrieval/hybrid_retriever.py`

BM25와 Dense는 점수 범위가 다르다. BM25 점수와 cosine 점수를 직접 더하면 안 되기 때문에 Reciprocal Rank Fusion, RRF를 사용했다.

RRF 방식:

```text
rrf_score = sum( weight / (rrf_k + rank) )
```

현재 설정:

| 항목 | 값 |
| --- | --- |
| `rrf_k` | 60 |
| `sparse_weight` | 1.0 |
| `dense_weight` | 1.0 |

검색 결과 schema:

| 필드 | 설명 |
| --- | --- |
| `doc_id` | 문서 ID |
| `bm25_rank` | BM25 순위 |
| `bm25_score` | BM25 점수 |
| `dense_rank` | Dense 순위 |
| `dense_score` | Dense 점수 |
| `rrf_score` | RRF 결합 점수 |
| `reranker_score` | reranker 점수 |
| `final_rank` | 최종 순위 |

## 12. Reranker 구현

구현 파일:

- `src/retrieval/reranker.py`

요청에서는 Cross-Encoder reranker를 요구했지만, 현재 MVP에서는 CPU-safe fallback을 구현했다.

현재 Reranker:

- `LexicalOverlapReranker`
- query token set과 document token set의 Jaccard overlap 점수 계산
- Hybrid RRF 상위 후보 30~50개를 다시 정렬
- 결과에 `reranker_score`와 `final_rank` 반환

아직 실제 `BAAI/bge-reranker-v2-m3` 모델은 연결하지 않았다. 구조상 이 파일만 교체하면 Cross-Encoder 기반 reranker로 확장할 수 있다.

## 13. 답변 생성

구현 파일:

- `src/generation/generator.py`

현재 답변 생성은 LLM을 호출하지 않는다. 대신 검색된 과거 상담 사례 중 최상위 문서의 `answer_original`을 그대로 근거 답변으로 반환한다.

현재 방식:

```text
사용자 질문
→ 검색 결과 Top-k
→ 1위 과거 상담 사례 선택
→ 해당 상담사 답변 반환
→ sources에 검색 근거 전체 표시
```

출력 구조:

| 필드 | 설명 |
| --- | --- |
| `answer` | 생성 또는 선택된 답변 |
| `grounded` | 검색 근거가 있는지 |
| `insufficient_context` | 근거 부족 여부 |
| `sources` | 사용한 과거 상담 문서 목록 |
| `retrieval_debug` | 검색/생성 디버그 정보 |

이 방식은 아직 자연스러운 생성형 답변은 아니지만, “검색된 상담 사례만 근거로 사용한다”는 원칙을 지키는 안전한 MVP이다.

## 14. Validation 평가 구성

구현 파일:

- `src/data/build_corpus.py`
- `src/evaluation/evaluate_retrieval.py`

중요한 점은 Validation 답변 문서를 Train 검색 corpus에 넣지 않았다는 것이다. 즉 validation 자체를 정답 문서로 검색하는 누수를 만들지 않았다.

대신 validation 질문에 대해 train 안의 같은 `카테고리 + 인텐트` 과거 상담 사례를 positive 문서로 연결했다.

positive 연결 순서:

1. 같은 카테고리 + 같은 인텐트의 Train 문서
2. 없으면 같은 인텐트의 Train 문서
3. 없으면 같은 카테고리의 Train 문서

이 방식은 “완전히 같은 질문의 정답 문서”를 찾는 평가가 아니라, “동일 업무 의도의 과거 상담 사례를 잘 찾는지”를 평가한다.

## 15. 검색 평가 지표

구현 파일:

- `src/evaluation/metrics.py`

평가 지표:

| 지표 | 의미 |
| --- | --- |
| `Hit@k` | positive 문서가 Top-k 안에 하나라도 있는지 |
| `Recall@k` | positive 문서 집합 중 Top-k에 포함된 비율 |
| `MRR` | 첫 positive 문서가 몇 번째에 등장했는지 |
| `nDCG@10` | 관련 문서의 순위 품질 |
| 평균 latency | 평균 검색 시간 |
| p95 latency | 95퍼센타일 검색 시간 |

생성 리포트:

- `reports/retrieval_metrics.csv`
- `reports/retrieval_metrics_summary.csv`
- `reports/retrieval_metrics.md`
- `reports/failure_cases.csv`
- `reports/category_metrics.csv`
- `reports/intent_metrics.csv`
- `reports/ann_benchmark.csv`

## 16. 현재 검색 성능

카페 카테고리 MVP 기준 실제 측정 결과는 다음과 같다.

| 방식 | Hit@5 | Recall@5 | MRR | nDCG@10 | 평균 지연시간(ms) | p95 지연시간(ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.4390 | 0.0130 | 0.3118 | 0.1627 | 8.09 | 18.22 |
| Dense Exact | 0.4100 | 0.0112 | 0.2814 | 0.1450 | 8.01 | 10.18 |
| Dense ANN | 0.3810 | 0.0124 | 0.2626 | 0.1498 | 53.18 | 59.28 |
| Hybrid RRF | 0.4340 | 0.0123 | 0.3075 | 0.1588 | 16.16 | 26.85 |
| Hybrid + Reranker | 0.4320 | 0.0123 | 0.3065 | 0.1573 | 22.61 | 33.23 |

해석:

- 현재 fallback 구현에서는 BM25가 가장 높은 Hit@5와 MRR을 보였다.
- Dense Exact는 빠르지만 BM25보다 낮았다.
- Dense ANN은 현재 SVD + sklearn 기반이라 정확도와 지연시간 모두 기대보다 좋지 않았다.
- Hybrid RRF는 BM25와 비슷했지만, 현재 Dense fallback이 약해서 큰 개선은 없었다.
- Reranker fallback은 실제 Cross-Encoder가 아니므로 성능 개선이 제한적이다.

## 17. API 구현

구현 파일:

- `src/api/main.py`

FastAPI 엔드포인트:

| API | 역할 |
| --- | --- |
| `GET /health` | 서버, 인덱스, 문서 수 상태 확인 |
| `POST /search` | 검색 결과와 점수 반환 |
| `POST /ask` | 검색 후 답변 생성 |
| `GET /metrics` | 최근 평가 summary 반환 |

실행 명령:

```powershell
uvicorn src.api.main:app --reload
```

## 18. UI 구현

구현 파일:

- `src/ui/app.py`

Streamlit UI 기능:

- 고객 질문 입력
- 검색 방식 선택
- Top-k 설정
- 카테고리 필터
- 인텐트 필터
- 생성 답변 표시
- 검색된 과거 상담 사례 표시
- 각 source의 질문, 답변, 점수 확인

실행 명령:

```powershell
streamlit run src/ui/app.py
```

## 19. 재현 명령

전체 MVP 재현 순서는 아래와 같다.

```powershell
python -m src.data.profile_data --config configs/mvp.yaml
python -m src.data.build_corpus --config configs/mvp.yaml
python -m src.retrieval.build_indexes --config configs/mvp.yaml
python -m src.evaluation.evaluate_retrieval --config configs/mvp.yaml
uvicorn src.api.main:app --reload
streamlit run src/ui/app.py
```

테스트:

```powershell
python -m pytest -q
```

현재 테스트 결과:

```text
3 passed
```

## 20. 현재 구현의 한계

현재 구현은 검색 RAG MVP이다. 다음 한계가 있다.

- Dense retrieval이 실제 딥러닝 임베딩이 아니라 TF-IDF fallback이다.
- ANN이 FAISS/HNSW가 아니라 sklearn fallback이다.
- Reranker가 Cross-Encoder가 아니라 lexical overlap fallback이다.
- `/ask`는 LLM 생성이 아니라 최상위 과거 답변을 반환한다.
- RAGAS 답변 품질 평가는 아직 없다.
- 비용, 토큰, LLM latency 모니터링은 아직 없다.
- 전체 카테고리 전체 데이터 확장은 아직 하지 않았다.

## 21. 다음 고도화 순서

추천 고도화 순서는 다음과 같다.

1. `BAAI/bge-m3` 임베딩 모델 연결
2. FAISS `IndexFlatIP` 기반 exact dense search 교체
3. FAISS HNSW 또는 IVF 기반 ANN 구현
4. `BAAI/bge-reranker-v2-m3` Cross-Encoder reranker 연결
5. 검색 성능 재평가
6. OpenAI 또는 로컬 LLM 기반 답변 생성 연결
7. 답변에 source doc_id와 과거 상담 근거 표시
8. RAGAS 기반 faithfulness, answer relevance 평가 추가
9. 비용, 토큰, latency logging 추가
10. 전체 카테고리 확장

## 22. 한 줄 요약

현재 구현은 “카페 카테고리 2만 개 train QA 문서와 2천 개 validation query를 이용해 과거 상담 사례를 검색하고, BM25/Dense/ANN/Hybrid/Reranker 방식별 검색 성능을 실제로 측정하는 RAG 검색 MVP”이다.

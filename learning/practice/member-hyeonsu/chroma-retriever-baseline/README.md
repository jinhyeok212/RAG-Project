# Chroma Retriever Baseline

KorQuAD 전용 구조에 묶이지 않고, 샘플 청크로 **질문 → 임베딩 → Chroma Top-k 검색 → JSONL 저장 → 시간 측정**을 연습하는 최소 구현입니다.

## 1. 먼저 알아둘 네 가지

- **Embedding(임베딩)**: 문장의 의미를 컴퓨터가 비교할 수 있는 숫자 벡터로 바꿉니다. 문서와 질문에 같은 모델을 써야 같은 공간에서 비교할 수 있습니다.
- **Chroma**: 청크의 벡터, 원문, 메타데이터를 저장하고 질문 벡터와 가까운 청크를 찾아주는 Vector DB입니다.
- **Retriever**: 질문을 임베딩하고 Vector DB에 검색을 요청한 뒤 관련 청크를 순서대로 돌려주는 RAG의 검색 부분입니다.
- **Top-k**: 가장 관련 있다고 판단한 결과를 몇 개 반환할지 뜻합니다. Top-1은 하나, Top-5는 다섯 개입니다. k가 크면 정답을 놓칠 가능성은 줄지만 불필요한 문맥이 늘 수 있습니다.

## 2. 폴더와 파일 구조

```text
chroma-retriever-baseline/
├─ config.json                 # DB/모델/거리/metadata 매핑/Top-k 설정
├─ requirements.txt            # 필요한 외부 라이브러리
├─ build_sample_db.py          # 개발·학습용 임시 Chroma DB 구축
├─ run_retrieval.py            # 실제 Retriever 실행 진입점
├─ data/
│  ├─ sample_chunks.jsonl      # 임시 DB 테스트용 샘플 청크 8개
│  └─ sample_questions.jsonl   # 테스트 질문 4개
├─ src/
│  ├─ config.py                # 설정 로드와 상대 경로 처리
│  ├─ embedding.py             # 텍스트 → 벡터
│  ├─ vector_store.py          # Chroma 저장/검색 API
│  └─ retriever.py             # 검색 흐름과 시간 측정
├─ vector_db/                  # 실행 시 생성됨(Git 제외)
└─ outputs/                    # 실행 시 생성됨(Git 제외)
```

파일을 나눈 이유는 각 부품을 따로 교체하기 위해서입니다. `build_sample_db.py`와 `sample_chunks.jsonl`은 앞 단계의 실제 Vector DB가 아직 없어서 만든 **개발·학습용 임시 도구**이며 최종 Retriever 산출물이 아닙니다. `run_retrieval.py`와 `src/`가 실제 검색 흐름입니다. 실제 Chroma DB를 받으면 임시 구축 스크립트는 실행하지 않고 설정을 맞춘 뒤 `run_retrieval.py`만 사용합니다. 복잡한 상속 구조나 프레임워크는 넣지 않았습니다.

## 3. 설치와 실행

Python 3.10~3.12 가상환경을 권장합니다. 아래 명령은 이 README가 있는 폴더에서 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python build_sample_db.py
python run_retrieval.py
```

`sentence-transformers`는 다국어 문장을 의미 벡터로 만들기 위해 사용합니다. 선택한 모델은 한국어를 포함한 다국어 baseline이며 첫 실행 시 모델 다운로드가 일어납니다. `chromadb`는 벡터와 메타데이터의 영구 저장 및 유사 벡터 탐색을 담당합니다.

Top-10을 추가하려면 `config.json`의 목록을 `[1, 3, 5, 10]`으로 바꾸거나 한 번만 다음처럼 실행합니다.

```powershell
python run_retrieval.py --top-k 1 3 5 10
```

## 4. 코드 실행 흐름

개발·학습 단계에서는 먼저 `build_sample_db.py`가 샘플 청크를 읽고, `TextEmbedder.embed()`로 벡터를 만든 뒤 `ChromaStore.upsert_chunks()`로 임시 DB에 저장합니다. 이 준비 단계는 실제 DB를 받으면 생략합니다.

실제 Retriever 실행 흐름은 다음과 같습니다.

1. `run_retrieval.py`가 설정과 질문을 읽습니다.
2. `Retriever.retrieve()`가 질문 임베딩 시간을 측정하면서 질문을 벡터로 바꿉니다.
3. `ChromaStore.query()`가 실제 또는 샘플 Chroma에서 가까운 벡터 Top-k를 찾고 DB 검색 시간을 측정합니다.
4. Retriever가 순위, 원본 거리, 계산 가능한 similarity, 시간 정보를 flat 레코드로 만듭니다.
5. `run_retrieval.py`가 검색 결과 한 건당 JSONL 한 줄로 저장합니다.

`perf_counter()`는 매우 짧은 실행 구간 측정에 적합한 단조 증가 고해상도 시계라 사용합니다. 시간 단위는 읽기 쉬운 밀리초(ms)입니다. 첫 질문은 모델 초기화 영향으로 이후 질문보다 느릴 수 있으므로 성능 실험에서는 워밍업 후 여러 번 측정하는 것이 좋습니다.

## 5. 주요 함수의 역할

- `load_settings()`: 설정 파일을 읽고 모든 상대 경로를 config 파일 위치 기준으로 해석합니다.
- `TextEmbedder.embed()`: 문서와 질문의 문자열 목록을 같은 임베딩 공간의 벡터로 변환합니다.
- `ChromaStore.upsert_chunks()`: `chunk_id`를 Chroma ID로 사용해 재실행해도 같은 청크를 갱신합니다.
- `ChromaStore.query()`: 질문 벡터로 Top-k 검색을 합니다. k가 저장 청크 수보다 크면 저장 개수까지만 반환합니다.
- `Retriever.retrieve()`: 질문 임베딩, 검색, 결과 변환과 세 구간의 시간 측정을 조율합니다.
- `write_jsonl()`: 검색 결과 한 건을 JSON 한 줄로 저장해 pandas나 평가 코드가 스트리밍으로 읽기 쉽게 합니다.

## 6. 정상 결과 예시

실행 뒤 다음 파일이 생기면 정상입니다.

```text
outputs/retrieval_top_1.jsonl   # 4개 질문 × 1개 결과 = 4줄
outputs/retrieval_top_3.jsonl   # 4개 질문 × 3개 결과 = 12줄
outputs/retrieval_top_5.jsonl   # 4개 질문 × 5개 결과 = 20줄
```

한 줄의 형태는 다음과 같습니다. 점수와 시간은 컴퓨터와 실행마다 달라집니다.

```json
{"question_id":"q-001","question":"출장이 끝난 뒤 정산서는 언제까지 내야 하나요?","top_k":3,"rank":1,"chunk_id":"chunk-004","document_id":"doc-expense","text":"출장비 정산서는 ... 7일 이내에 ... 제출합니다.","distance":0.22,"similarity":0.78,"embedding_time_ms":14.3,"search_time_ms":2.1,"total_time_ms":16.4}
```

`distance`는 Chroma가 반환한 원본 값을 거리 방식과 관계없이 항상 저장합니다. `similarity = 1 - distance`라는 변환은 **cosine distance일 때만** 의미가 맞으므로 `config.json`의 `distance_metric`이 `cosine`일 때만 계산합니다. 이 경우 distance는 작을수록, similarity는 클수록 가깝습니다. `l2` 또는 `ip`라면 임의 변환하지 않고 `similarity`를 JSON `null`로 저장합니다. 필드를 `null`로 유지하는 이유는 모든 행이 같은 스키마를 갖게 하기 위해서입니다.

각 검색 결과 행에는 평가 준비에 필요한 다음 필드가 항상 있습니다.

```text
question_id, question, top_k, rank, chunk_id, document_id,
distance, similarity, embedding_time_ms, search_time_ms, total_time_ms
```

`similarity`는 계산할 수 없는 거리 방식에서는 `null`입니다. `text`도 검색 내용 확인을 위해 추가로 저장합니다. 이 flat 구조는 pandas의 `groupby(["question_id", "top_k"])`로 질문과 k별 결과를 묶기 쉽습니다. 나중에 정답 라벨을 별도 데이터와 `question_id`로 병합하면 정답이 Top-k 안에 있는지로 Hit@k를, 최초 정답의 `rank`로 reciprocal rank를 구해 MRR을 계산할 수 있습니다.

현재는 정답 청크 기준이 확정되지 않았으므로 `gold_chunk_id`, `gold_document_id`, `is_correct`를 강제로 넣지 않습니다. 이후 라벨이 확정되면 검색 코드에 결합하기보다, 우선 결과 JSONL과 정답 파일을 `question_id`로 병합하는 별도 평가 스크립트를 추가하는 방식이 간단합니다. 필요해지면 질문 JSONL의 선택 필드를 결과에 전달하도록 `run_retrieval.py`를 확장할 수도 있으며 현재 Retriever의 검색 로직은 바꿀 필요가 없습니다.

각 결과 행에 같은 질문의 시간이 반복되는 것은 해당 검색 실행의 시간임을 명확히 남기기 위해서입니다. `total_time_ms`는 임베딩과 DB 검색 외에 결과 조립에 드는 아주 작은 오버헤드도 포함합니다.

## 7. 데이터 형식과 실제 데이터로 교체하기

샘플 청크는 다음 최소 규칙만 사용하므로 KorQuAD 구조에 의존하지 않습니다.

```json
{"chunk_id":"고유 청크 ID","document_id":"원문 문서 ID","text":"검색할 청크 원문","metadata":{"title":"선택 정보"}}
```

이 형식은 임시 샘플 DB를 직접 다시 만들 때만 필요합니다. `metadata`에는 Chroma가 지원하는 단순 값(문자열, 숫자, 불리언)을 넣는 편이 안전합니다. 실제 DB를 전달받는 흐름에서는 샘플 JSONL이나 `chunks_path`가 Retriever 실행에 사용되지 않습니다.

완성된 Chroma DB를 직접 받는 경우에는 다음을 확인합니다.

1. `config.json`의 `db_path`와 `collection_name`을 실제 값으로 변경합니다.
2. `embedding_model`을 **그 DB의 문서를 임베딩할 때 쓴 모델과 정확히 동일하게** 변경합니다. 모델이 다르면 벡터 차원이 같더라도 검색 의미가 맞지 않습니다.
3. `distance_metric`을 컬렉션 구축 시 사용한 `cosine`, `l2`, `ip` 중 실제 값으로 맞춥니다. 이 값은 similarity 계산 여부에도 사용됩니다.
4. 실제 metadata 키가 다르면 코드 대신 `metadata_fields`만 바꿉니다. 예를 들어 실제 키가 `chunk_no`, `source_id`라면 `{"chunk_id": "chunk_no", "document_id": "source_id"}`로 설정합니다.
5. 이미 구축된 DB이므로 `build_sample_db.py`는 실행하지 않고 `run_retrieval.py`만 실행합니다.
6. DB가 서버형 Chroma라면 연결 방식 자체가 달라지므로 `vector_store.py`의 `PersistentClient`만 `HttpClient`로 교체합니다. Retriever 로직은 그대로 유지됩니다.

주의: 기존 컬렉션은 생성 시 정한 거리 방식과 임베딩 차원을 유지합니다. 다른 모델이나 거리 방식으로 실험하려면 새 collection 이름을 사용하는 것이 안전합니다.

## 8. 다음 실험으로 확장할 지점

이 baseline이 동작하고 정답 기준이 확정된 뒤에는 별도 정답 파일을 결과와 결합해 Hit@k/Recall@k/MRR을 계산하고, 여러 임베딩 모델·chunk 크기·거리 방식을 같은 질문 세트로 비교할 수 있습니다. 현재 코드는 검색 구조 학습에 집중하여 정답 필드, 평가 지표, 재정렬(reranking)은 의도적으로 포함하지 않았습니다.

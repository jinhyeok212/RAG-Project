# Experiment Configuration Contract v1.0

## 1. 문서 목적

이 문서는 RAG 실험 자동화에서 사용하는 공통 YAML 설정의 필드와 의미를 정의한다.

데이터 검증, 청킹, 임베딩, 인덱싱, 검색, 평가 모듈이 서로 다른 경로나 설정값을 사용하지 않도록 모든 실험 조건을 하나의 YAML 파일로 관리한다.

Experiment Runner는 다음 설정 파일을 읽어 전체 실험을 실행한다.

```text
configs/experiments/exp_full_kure_v1.yaml
```

자동화 흐름은 다음과 같다.

```text
실험 YAML 검증
→ 데이터 검증
→ 청킹 및 Chunk qrels 검증
→ 인덱스 생성 또는 재사용
→ 전체 평가 질문 검색
→ Hit@1·3·5 및 MRR 계산
→ 실험 결과 저장
```

---

## 2. 공통 실험 설정

```yaml
schema_version: "1.0"

experiment_id: exp_full_kure_v1
experiment_name: full-dataset-kure-baseline
base_experiment_id: null

dataset:
  version: ontong_youth_full_v1
  preprocessing_version: ontong_youth_document_v1
  document_path: data/processed/ontong_youth_full_v1/documents.jsonl
  chunk_path: data/processed/ontong_youth_full_v1/chunks.jsonl
  manifest_path: data/processed/ontong_youth_full_v1/dataset_manifest.json

evaluation:
  version: eval_full_v1
  questions_path: data/evaluation/eval_full_v1/eval_questions.jsonl
  document_qrels_path: data/evaluation/eval_full_v1/qrels.jsonl
  chunk_qrels_path: data/evaluation/eval_full_v1/qrels_chunk.jsonl
  manifest_path: data/evaluation/eval_full_v1/evaluation_manifest.json

chunking:
  version: c2_section_800_ov120_v1
  strategy: section
  size: 800
  overlap: 120
  unit: characters
  target_field: retrieval_text

embedding:
  model: nlpai-lab/KURE-v1
  dimension: 1024
  normalize: true
  batch_size: 32
  device: auto

index:
  version: index_full_kure_v1
  db_path: indexes/chroma/index_full_kure_v1
  collection_name: ontong_youth_full_kure_v1
  distance_metric: cosine

retrieval:
  evaluation_max_k: 5
  context_top_k: 3

runtime:
  output_root: experiments
  registry_path: registry/experiments
  fail_fast: true

pipeline:
  entrypoints:
    validate_documents: src.data.validate_documents:validate_documents
    build_chunks: src.chunking.build_chunks:build_chunks
    resolve_index: src.indexing.index_resolver:resolve_index
    run_batch_retrieval: src.retrieval.batch_retriever:run_batch_retrieval
    evaluate_experiment: src.evaluation.evaluate_experiment:evaluate_experiment
    write_experiment_artifacts: src.experiments.artifact_writer:write_experiment_artifacts
```

---

## 3. 기본 정보

```yaml
schema_version: "1.0"

experiment_id: exp_full_kure_v1
experiment_name: full-dataset-kure-baseline
base_experiment_id: null
```

### `schema_version`

공통 실험 YAML 규격의 버전이다.

현재 최초 규격은 `1.0`을 사용한다. 필드가 추가되거나 기존 필드의 의미가 변경되면 Schema 버전을 함께 변경한다.

### `experiment_id`

실험을 구분하는 고유 ID다.

```yaml
experiment_id: exp_full_kure_v1
```

실험 결과는 다음 경로에 저장된다.

```text
experiments/exp_full_kure_v1/
```

동일한 ID를 반복 사용하면 이전 결과와 구분하기 어려우므로, 실험 조건이 달라지면 새로운 ID를 사용한다.

권장 형식은 다음과 같다.

```text
exp_<dataset>_<model>_<version>
```

예시:

```text
exp_full_kure_v1
exp_full_bge_m3_v1
exp_full_model_a_v1
```

### `experiment_name`

사람이 실험 목적을 쉽게 이해할 수 있도록 작성하는 이름이다.

```yaml
experiment_name: full-dataset-kure-baseline
```

`experiment_id`는 폴더 및 시스템 식별에 사용하고, `experiment_name`은 결과 보고서와 대시보드 표시용으로 사용한다.

### `base_experiment_id`

현재 실험과 비교할 기준 실험의 ID다.

```yaml
base_experiment_id: null
```

`exp_full_kure_v1` 자체가 전체 데이터 KURE-v1 Baseline이므로 비교 대상이 없어 `null`로 설정한다.

다른 Embedding 모델을 비교할 때는 다음과 같이 설정한다.

```yaml
experiment_id: exp_full_bge_m3_v1
base_experiment_id: exp_full_kure_v1
```

이 경우 `exp_full_bge_m3_v1`의 결과를 KURE-v1 Baseline과 비교한다.

---

## 4. Dataset 설정

```yaml
dataset:
  version: ontong_youth_full_v1
  preprocessing_version: ontong_youth_document_v1
  document_path: data/processed/ontong_youth_full_v1/documents.jsonl
  chunk_path: data/processed/ontong_youth_full_v1/chunks.jsonl
  manifest_path: data/processed/ontong_youth_full_v1/dataset_manifest.json
```

온통청년 전체 데이터의 버전과 파일 위치를 정의한다.

### `version`

실험에 사용하는 Dataset 버전이다.

```yaml
version: ontong_youth_full_v1
```

원본 데이터의 범위나 구성이 달라지면 새로운 버전을 사용한다.

### `preprocessing_version`

원본 데이터를 공통 Document 형식으로 변환한 전처리 규칙의 버전이다.

```yaml
preprocessing_version: ontong_youth_document_v1
```

정제 규칙, 필드 매핑 또는 누락값 처리 방식이 달라지면 새로운 전처리 버전을 사용한다.

### `document_path`

전처리가 완료된 전체 정책 문서의 위치다.

```yaml
document_path: data/processed/ontong_youth_full_v1/documents.jsonl
```

각 문서는 최소한 다음 필드를 가져야 한다.

```json
{
  "document_id": "policy_0001",
  "title": "정책 제목",
  "retrieval_text": "검색에 사용할 정책 내용",
  "category": "일자리",
  "source": "ontong_youth"
}
```

### `chunk_path`

검색과 임베딩에 사용할 청크 파일의 위치다.

```yaml
chunk_path: data/processed/ontong_youth_full_v1/chunks.jsonl
```

각 청크는 최소한 다음 필드를 가져야 한다.

```json
{
  "chunk_id": "policy_0001_chunk_001",
  "document_id": "policy_0001",
  "text": "검색에 사용할 청크 내용"
}
```

### `manifest_path`

Dataset의 생성 정보와 무결성 정보를 기록한 Manifest 위치다.

```yaml
manifest_path: data/processed/ontong_youth_full_v1/dataset_manifest.json
```

Manifest에는 최소한 다음 정보가 포함되어야 한다.

* Dataset Version
* Preprocessing Version
* 문서 수
* 청크 수
* 데이터 파일 SHA256
* 생성 시각
* 생성 코드 또는 규칙 버전

### 현재 저장소 상태

현재 저장소의 `chunking2` 자료는 온통청년 정책 400개를 사용한 MVP 데이터다.

기존 Document Manifest:

```text
data/processed/chunking2/04.manifests/document_manifest.json
```

새로운 YAML에 정의한 `ontong_youth_full_v1` 경로는 앞으로 전체 데이터 담당자가 생성할 목표 경로다.

현재 해당 경로에 파일이 없더라도 MVP 경로로 임의 변경하지 않는다.

---

## 5. Evaluation 설정

```yaml
evaluation:
  version: eval_full_v1
  questions_path: data/evaluation/eval_full_v1/eval_questions.jsonl
  document_qrels_path: data/evaluation/eval_full_v1/qrels.jsonl
  chunk_qrels_path: data/evaluation/eval_full_v1/qrels_chunk.jsonl
  manifest_path: data/evaluation/eval_full_v1/evaluation_manifest.json
```

평가 질문과 정답 데이터의 버전 및 위치를 정의한다.

### `version`

평가셋의 버전이다.

```yaml
version: eval_full_v1
```

평가 질문이 추가·수정되거나 정답 qrels가 변경되면 평가셋 버전도 변경한다.

### `questions_path`

평가 질문 목록의 위치다.

```yaml
questions_path: data/evaluation/eval_full_v1/eval_questions.jsonl
```

예시:

```json
{
  "query_id": "q001",
  "query": "청년 월세 지원을 받을 수 있는 조건은 무엇인가요?"
}
```

### `document_qrels_path`

질문별 정답 문서 ID를 기록한 파일의 위치다.

```yaml
document_qrels_path: data/evaluation/eval_full_v1/qrels.jsonl
```

예시:

```json
{
  "query_id": "q001",
  "document_id": "policy_0001",
  "relevance": 1
}
```

### `chunk_qrels_path`

질문별 정답 청크 ID를 기록한 파일의 위치다.

```yaml
chunk_qrels_path: data/evaluation/eval_full_v1/qrels_chunk.jsonl
```

예시:

```json
{
  "query_id": "q001",
  "chunk_id": "policy_0001_chunk_001",
  "relevance": 1
}
```

### `manifest_path`

평가셋의 생성 및 검수 정보를 기록한 Manifest 위치다.

```yaml
manifest_path: data/evaluation/eval_full_v1/evaluation_manifest.json
```

Manifest에는 최소한 다음 정보가 포함되어야 한다.

* Evaluation Version
* 전체 질문 수
* 문서 qrels 수
* Chunk qrels 수
* 질문·qrels 파일 SHA256
* 생성 및 검수 기준
* 생성 시각

모든 평가 질문은 문서 qrels와 연결되어야 한다. Chunk 단위 평가를 진행하려면 Chunk qrels도 실제 `chunks.jsonl`의 `chunk_id`와 연결되어야 한다.

---

## 6. Chunking 설정

```yaml
chunking:
  version: c2_section_800_ov120_v1
  strategy: section
  size: 800
  overlap: 120
  unit: characters
  target_field: retrieval_text
```

문서를 검색용 청크로 분리하는 조건을 정의한다.

### `version`

청킹 설정의 버전이다.

```yaml
version: c2_section_800_ov120_v1
```

버전명에는 청킹 방식, 크기, Overlap, 버전을 포함한다.

```text
<전략>_<크기>_ov<중첩>_<version>
```

### `strategy`

청킹 방식을 정의한다.

```yaml
strategy: section
```

현재 Baseline은 정책 문서의 항목 구조를 활용하는 Section 기반 청킹을 사용한다.

### `size`

청크 본문의 최대 크기다.

```yaml
size: 800
```

현재 단위가 `characters`이므로 최대 800자를 의미한다.

### `overlap`

긴 Section을 여러 청크로 분리할 때 인접한 청크가 공유하는 문자 수다.

```yaml
overlap: 120
```

Overlap은 Chunk Size보다 작아야 한다.

```text
0 ≤ overlap < size
```

현재 설정에서는 다음 조건을 만족한다.

```text
0 ≤ 120 < 800
```

### `unit`

Chunk Size와 Overlap의 측정 단위다.

```yaml
unit: characters
```

현재 프로젝트는 문자 수를 기준으로 청킹한다.

### `target_field`

청킹 대상이 되는 Document 필드다.

```yaml
target_field: retrieval_text
```

원본 데이터의 모든 필드를 무조건 합치는 것이 아니라, 전처리 과정에서 검색용으로 구성한 `retrieval_text`를 청킹한다.

### 현재 설정 근거

현재 MVP Chunk Manifest에도 다음 조건이 기록되어 있다.

```text
max_body_chars: 800
overlap_chars: 120
```

기존 Manifest 위치:

```text
data/processed/chunking2/05.chunks/c2_section_800/c2_section_800_manifest.json
```

---

## 7. Embedding 설정

```yaml
embedding:
  model: nlpai-lab/KURE-v1
  dimension: 1024
  normalize: true
  batch_size: 32
  device: auto
```

청크와 평가 질문을 벡터로 변환할 때 사용하는 설정이다.

### `model`

사용할 Embedding 모델의 정확한 식별자다.

```yaml
model: nlpai-lab/KURE-v1
```

모델의 약칭이 아닌 실제 코드에서 불러올 수 있는 Hugging Face 모델 ID를 기록한다.

### `dimension`

Embedding Vector의 차원이다.

```yaml
dimension: 1024
```

인덱스 생성 후 실제 모델이 반환한 차원과 YAML 설정값이 일치하는지 검증해야 한다.

### `normalize`

Embedding Vector의 L2 정규화 여부다.

```yaml
normalize: true
```

문서와 질문 Embedding에 동일한 정규화 조건을 적용한다.

### `batch_size`

한 번의 Embedding 연산에서 처리할 입력 수다.

```yaml
batch_size: 32
```

GPU 메모리가 부족하면 32에서 16 또는 8로 낮출 수 있다. Batch Size를 변경하더라도 동일한 모델과 정규화 설정을 사용한다면 검색 결과의 의미는 유지되어야 한다.

### `device`

Embedding 연산에 사용할 장치를 지정한다.

```yaml
device: auto
```

`auto`는 실행 환경에 따라 사용 가능한 장치를 자동으로 선택한다.

```text
CUDA 사용 가능 → cuda
Apple Silicon MPS 사용 가능 → mps
그 외 → cpu
```

현재 `src/rag/embedding.py`에도 KURE-v1 모델, 정규화, Batch Size 및 장치 선택 기능이 구현되어 있다.

---

## 8. Index 설정

```yaml
index:
  version: index_full_kure_v1
  db_path: indexes/chroma/index_full_kure_v1
  collection_name: ontong_youth_full_kure_v1
  distance_metric: cosine
```

ChromaDB 인덱스의 버전과 저장 위치를 정의한다.

### `version`

인덱스 버전이다.

```yaml
version: index_full_kure_v1
```

다음 조건 중 하나라도 변경되면 새로운 인덱스 버전을 사용한다.

* Dataset Version
* Preprocessing Version
* Chunking Version
* Chunk Size
* Chunk Overlap
* Embedding Model
* Embedding Dimension
* Embedding Normalization
* Distance Metric

### `db_path`

ChromaDB Persistent Index가 저장되는 경로다.

```yaml
db_path: indexes/chroma/index_full_kure_v1
```

기존 MVP 인덱스를 보호하기 위해 전체 데이터 인덱스는 별도 경로에 생성한다.

### `collection_name`

ChromaDB Collection의 이름이다.

```yaml
collection_name: ontong_youth_full_kure_v1
```

Collection 이름은 Dataset과 Embedding 모델을 구분할 수 있도록 작성한다.

기존 Collection을 삭제하거나 덮어쓰지 않는다. 설정이 달라지면 새로운 Collection을 생성한다.

### `distance_metric`

Vector 검색에서 사용할 거리 함수다.

```yaml
distance_metric: cosine
```

현재 KURE-v1 인덱스는 Cosine Distance를 사용한다.

검색 결과에는 원본 `distance`를 보존하고, 필요하면 다음과 같이 Similarity를 계산한다.

```text
similarity = 1 - cosine_distance
```

---

## 9. Retrieval 설정

```yaml
retrieval:
  evaluation_max_k: 5
  context_top_k: 3
```

검색 평가와 답변 생성에서 사용할 Top-k를 정의한다.

### `evaluation_max_k`

평가 과정에서 검색할 최대 결과 수다.

```yaml
evaluation_max_k: 5
```

현재 핵심 지표가 Hit@1, Hit@3, Hit@5이므로 최대 5개까지 검색한다.

한 번의 Top-5 검색 결과를 이용해 다음 지표를 모두 계산할 수 있다.

```text
Hit@1
Hit@3
Hit@5
MRR
```

### `context_top_k`

실제 답변 생성 단계에서 사용할 검색 결과 수다.

```yaml
context_top_k: 3
```

평가용 최대 검색 범위와 답변 Context 수는 목적이 다르므로 별도로 관리한다.

```text
evaluation_max_k = 검색 성능 평가 범위
context_top_k = 답변 생성에 전달할 청크 수
```

다음 조건을 유지해야 한다.

```text
context_top_k ≤ evaluation_max_k
```

현재 설정은 다음과 같다.

```text
3 ≤ 5
```

---

## 10. Runtime 설정

```yaml
runtime:
  output_root: experiments
  registry_path: registry/experiments
  fail_fast: true
```

실험 결과 저장 위치와 실행 실패 처리 방식을 정의한다.

### `output_root`

실험 결과가 저장될 최상위 폴더다.

```yaml
output_root: experiments
```

실제 결과는 `experiment_id`를 하위 폴더명으로 사용한다.

```text
experiments/
└─ exp_full_kure_v1/
   ├─ experiment_manifest.json
   ├─ index_manifest.json
   ├─ metrics_summary.json
   ├─ metrics_by_query.jsonl
   ├─ retrieval_results.jsonl
   ├─ traces.jsonl
   ├─ failure_cases.csv
   └─ run.log
```

`configs/experiments/`와 루트의 `experiments/`는 역할이 다르다.

```text
configs/experiments/
= 실험 실행 전에 사람이 작성하는 설정

experiments/
= 실험 실행 후 Runner가 저장하는 결과
```

### `registry_path`

실행한 실험의 상태와 결과 위치를 기록할 Registry 경로다.

```yaml
registry_path: registry/experiments
```

Registry는 이후 Experiment Registry 구현 단계에서 사용한다.

### `fail_fast`

중간 단계에서 오류가 발생하면 즉시 실행을 중단할지 결정한다.

```yaml
fail_fast: true
```

다음과 같은 문제가 발생하면 이후 단계로 넘어가지 않는다.

* YAML Schema 오류
* 필수 데이터 파일 누락
* JSONL 파싱 오류
* 중복 Document 또는 Chunk ID
* 질문과 qrels 연결 누락
* 존재하지 않는 정답 Chunk ID
* Embedding 차원 불일치
* Index Manifest 불일치

잘못된 입력으로 임베딩과 인덱스 생성을 진행하면 실행 시간과 저장 공간을 낭비할 수 있으므로 현재는 `true`로 고정한다.

---

## 11. Pipeline Entrypoint

```yaml
pipeline:
  entrypoints:
    validate_documents: src.data.validate_documents:validate_documents
    build_chunks: src.chunking.build_chunks:build_chunks
    resolve_index: src.indexing.index_resolver:resolve_index
    run_batch_retrieval: src.retrieval.batch_retriever:run_batch_retrieval
    evaluate_experiment: src.evaluation.evaluate_experiment:evaluate_experiment
    write_experiment_artifacts: src.experiments.artifact_writer:write_experiment_artifacts
```

Experiment Runner가 각 단계에서 호출할 Python 함수의 위치를 정의한다.

Entrypoint는 다음 형식을 사용한다.

```text
파이썬 모듈 경로:함수 이름
```

예시:

```text
src.data.validate_documents:validate_documents
```

Runner에서는 다음 함수를 불러온다는 의미다.

```python
from src.data.validate_documents import validate_documents
```

### 단계별 Entrypoint

| 순서 | Entrypoint                   | 역할                      |
| -: | ---------------------------- | ----------------------- |
|  1 | `validate_documents`         | 문서와 Dataset Manifest 검증 |
|  2 | `build_chunks`               | 청킹 실행 및 Chunk qrels 검증  |
|  3 | `resolve_index`              | Chroma 인덱스 생성 또는 재사용    |
|  4 | `run_batch_retrieval`        | 전체 평가 질문 Top-k 검색       |
|  5 | `evaluate_experiment`        | Hit@k, MRR 및 실패 사례 계산   |
|  6 | `write_experiment_artifacts` | 표준 실험 결과 파일 저장          |

현재 일부 모듈이 아직 존재하지 않는 것은 정상이다.

PM이 모듈 위치와 함수 이름을 먼저 확정하고, 각 담당자가 해당 계약에 맞춰 구현한다.

팀원은 내부 구현 방식을 자유롭게 결정할 수 있지만 다음 항목은 임의로 변경하지 않는다.

* 함수 이름
* 함수 인자 순서
* 공통 반환 필드
* YAML 필드명
* 공통 파일 경로

변경이 필요하면 먼저 팀에 공유하고 YAML, JSON Schema, 계약 문서를 함께 수정한다.

---

## 12. Embedding 모델 비교 규칙

Embedding 모델을 비교할 때는 다른 실험 조건을 동일하게 유지한다.

고정 항목:

* Dataset Version
* Preprocessing Version
* 평가 질문
* 문서 qrels
* Chunk qrels
* Chunking Version
* Chunk Size
* Chunk Overlap
* Distance Metric
* `evaluation_max_k`

변경 항목:

* `experiment_id`
* `experiment_name`
* `base_experiment_id`
* `embedding.model`
* `embedding.dimension`
* `index.version`
* `index.db_path`
* `index.collection_name`

예시:

```yaml
experiment_id: exp_full_bge_m3_v1
experiment_name: full-dataset-bge-m3
base_experiment_id: exp_full_kure_v1

embedding:
  model: BAAI/bge-m3
  dimension: 1024
  normalize: true
  batch_size: 32
  device: auto

index:
  version: index_full_bge_m3_v1
  db_path: indexes/chroma/index_full_bge_m3_v1
  collection_name: ontong_youth_full_bge_m3_v1
  distance_metric: cosine
```

비교 모델의 정확한 설정값은 실제 실험 전에 모델 공식 문서와 구현 결과를 확인한 뒤 확정한다.

---

## 13. 팀 공통 규칙

1. 실험 설정값과 경로를 코드에 하드코딩하지 않는다.
2. 모든 모듈은 공통 YAML에서 설정을 읽는다.
3. 공통 필드명을 개인 판단으로 변경하지 않는다.
4. 기존 ChromaDB Collection을 덮어쓰지 않는다.
5. 기존 실험 결과 폴더를 덮어쓰지 않는다.
6. 데이터 또는 qrels 검증에 실패하면 다음 단계로 넘어가지 않는다.
7. 실제 측정값과 예시값을 구분해서 기록한다.
8. 설정을 변경하면 실험 ID와 관련 Version을 함께 변경한다.
9. 모듈의 입출력 형식을 변경할 경우 PM에게 먼저 공유한다.
10. YAML, JSON Schema, 계약 문서의 내용이 서로 일치하도록 관리한다.

---

## 14. 현재 단계

현재 단계에서는 다음 작업을 우선 진행한다.

1. 공통 실험 YAML 확정
2. `experiment.schema.json` 작성
3. 모듈별 입력·출력 계약 확정
4. 담당자별 모듈 구현
5. `runner.py`에서 전체 모듈 연결
6. 전체 데이터 KURE-v1 Baseline 실행

Quality Gate의 구체적인 수치, Release, Staging, Production 배포 및 Rollback은 전체 데이터 Baseline 결과가 생성된 후 진행한다.

---

## 15. Experiment Schema

공통 실험 YAML은 다음 JSON Schema로 검증한다.

```text
schemas/experiment.schema.json

---

## 16. Config Validator

공통 실험 YAML은 다음 모듈에서 자동으로 검증한다.

```text
src/experiments/config_validator.py

---

## 17. 모듈 입출력 계약

각 Pipeline 모듈은 다음 함수명과 인자 순서를 사용한다.

```python
def validate_documents(config) -> DatasetInfo:
    ...

def build_chunks(config, dataset_info) -> ChunkInfo:
    ...

def resolve_index(config, chunk_info) -> IndexInfo:
    ...

def run_batch_retrieval(
    config,
    index_info,
) -> list[RetrievalResult]:
    ...

def evaluate_experiment(
    config,
    retrieval_results,
) -> EvaluationResult:
    ...

def write_experiment_artifacts(
    config,
    run_context,
) -> ArtifactInfo | None:
    ...

---

## 18. Experiment Runner

전체 실험 실행은 다음 모듈이 담당한다.

```text
src/experiments/runner.py

---

## 19. Experiment Runner 테스트

Experiment Config, Module Contract 및 Runner 연결 구조는 다음 테스트에서 검증한다.

```text
tests/test_experiment_runner.py
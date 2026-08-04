# RAG 공부 프로젝트

AIHub 기계독해 데이터를 사용해서 RAG의 검색 단계를 이해하기 위한 실험 프로젝트입니다.

## 목표

처음 목표는 LLM 없이 Retrieval-only RAG를 구현하는 것입니다.

```text
AIHub 데이터
-> context / question / answer 분리
-> context 청킹
-> 임베딩
-> FAISS 벡터 검색
-> 검색 결과에 answer가 포함되는지 평가
```

## 폴더 구조

```text
Rag공부/
├─ Data/
│  ├─ 기계독해/          # AIHub 원본 데이터
│  ├─ processed/         # 전처리 결과
│  └─ samples/           # 소량 실험 샘플
│
├─ vector_store/
│  └─ faiss/             # FAISS 인덱스와 메타데이터
│
├─ outputs/              # 검색 결과, 평가 결과, 실패 사례
├─ notebooks/            # 스터디용 노트북
├─ docs/                 # 단계별 설명 문서
├─ ui/                   # 결과 확인용 대시보드
├─ src/                  # 단계별 Python 코드
├─ requirements.txt
└─ README.md
```

## 1차 구현 범위

- 데이터: AIHub 기계독해 `01.Normal.zip` 또는 압축 해제된 `ko_nia_normal_squad_all.json`
- 임베딩 모델: `intfloat/multilingual-e5-small`
- 벡터DB: FAISS
- 실험:
  - 청크 크기 300 vs 700
  - Top-k 3 vs 5
  - 검색 결과에 정답 포함 여부 확인

## 실행 순서

### 1. 샘플 생성

```bash
python src/01_build_sample.py
```

생성 파일:

```text
Data/samples/documents.jsonl
Data/samples/eval_questions.csv
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. FAISS 인덱스 생성

```bash
python src/02_build_index.py --chunk-size 300
python src/02_build_index.py --chunk-size 700
```

### 4. 검색 평가

```bash
python src/03_evaluate_retrieval.py --index-dir vector_store/faiss/chunk_300 --top-k 3
python src/03_evaluate_retrieval.py --index-dir vector_store/faiss/chunk_300 --top-k 5
python src/03_evaluate_retrieval.py --index-dir vector_store/faiss/chunk_700 --top-k 3
python src/03_evaluate_retrieval.py --index-dir vector_store/faiss/chunk_700 --top-k 5
```

### 5. 대시보드 생성

```bash
python src/04_generate_dashboard.py
```

생성 파일:

```text
ui/index.html
```

### 6. 채팅형 UI 실행

ChatGPT처럼 질문을 입력하고 검색 결과를 답변 형태로 확인하려면 다음을 실행합니다.

```bash
python src/05_chat_app.py
```

브라우저에서 접속:

```text
http://127.0.0.1:8788/
```

현재 채팅 UI는 Retrieval-only입니다. LLM을 붙이지 않았기 때문에 새 답변을 생성하지 않고, 질문과 가장 가까운 문서 청크를 근거로 보여줍니다.

채팅 UI에서 질문을 입력하면 응답 시간이 함께 표시되고, 실행 로그는 다음 파일에 누적 저장됩니다.

```text
outputs/chat_logs.csv
```

## 설명 문서

- [RAG 전체 흐름](docs/00_RAG_전체흐름.md)
- [샘플 생성 설명](docs/01_샘플생성_설명.md)
- [FAISS 인덱스 설명](docs/02_FAISS인덱스_설명.md)
- [검색 평가 설명](docs/03_검색평가_설명.md)
- [스터디 숙제 정리](docs/04_스터디_숙제_정리.md)

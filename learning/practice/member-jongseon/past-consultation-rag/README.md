# 과거 상담 사례 검색 기반 RAG 연습

소상공인 고객 주문 질의-응답 상담 로그를 이용해 과거 상담 QA 문서를 복원하고, 사용자 질문과 유사한 과거 상담 사례를 검색해 답변 근거로 사용하는 RAG MVP입니다.

## 범위

- 상담 발화 CSV에서 질문-답변 QA 문서 재구성
- 개인정보 추정 패턴 마스킹
- BM25 sparse retrieval
- TF-IDF 기반 Dense retrieval fallback
- SVD + NearestNeighbors 기반 ANN fallback
- Hybrid RRF 검색
- Lexical overlap reranker fallback
- FastAPI API
- Streamlit UI
- Validation query 기반 retrieval 평가

현재 버전은 학습용 baseline입니다. Sentence-Transformers, BAAI/bge-m3, Cross-Encoder reranker, RAGAS 평가는 아직 적용하지 않았습니다.

## 데이터

원본 상담 데이터와 생성된 index/vector DB는 GitHub에 포함하지 않습니다.

로컬 실행 시 아래 구조로 원본 CSV를 배치해야 합니다.

```text
소상공인 고객 주문 질의-응답 텍스트/
├── Training/
└── Validation/
```

기본 설정은 [configs/mvp.yaml](configs/mvp.yaml)에 있습니다.

## 실행 순서

```powershell
pip install -r requirements.txt

python -m src.data.profile_data --config configs/mvp.yaml
python -m src.data.build_corpus --config configs/mvp.yaml
python -m src.retrieval.build_indexes --config configs/mvp.yaml
python -m src.evaluation.evaluate_retrieval --config configs/mvp.yaml
```

## API 실행

```powershell
pip install fastapi uvicorn
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

주요 엔드포인트:

- `GET /health`
- `POST /search`
- `POST /ask`
- `GET /metrics`

## UI 실행

```powershell
pip install streamlit
python -m streamlit run src/ui/app.py --server.address 127.0.0.1 --server.port 8501
```

## 테스트

```powershell
python -m pytest -q
```

## 업로드 제외 항목

- `.env`, API key, token, password
- 원본 상담 CSV
- KorQuAD 원본/가공 대용량 데이터
- `data/indexes`의 `.pkl`, `.joblib` index 파일
- `data/processed`의 parquet/jsonl/csv 산출물
- `reports` 실행 결과
- `__pycache__`, `.pytest_cache`
- PDF/이미지 보고서 산출물

## 참고 문서

- [구현 상세 보고서](docs/RAG_구현_상세_보고서.md)
- [발표대본](docs/RAG_구현_상세_발표대본.md)

# RAG Core Chatbot for Team Practice

팀원들이 본격적인 RAG Quality Monitor 프로젝트를 시작하기 전에, RAG의 핵심을 직접 이해하기 위한 교육용 미니 챗봇입니다.

이 앱은 한국어 지식베이스 문서를 기반으로 다음 과정을 보여줍니다.

1. 문서 로딩
2. 문서 chunking
3. 검색 인덱스 생성
4. 사용자 질문 검색
5. 프롬프트 구성
6. 답변 생성
7. 실행 로그 저장
8. 간단한 로그 대시보드 확인

## 1. 이 챗봇이 답할 수 있는 질문

기본 지식베이스에는 RAG 핵심 개념이 들어 있습니다.

예시 질문:

- RAG가 뭐야?
- RAG에는 어떤 데이터가 필요해?
- chunk_size를 크게 하면 어떤 문제가 생겨?
- top_k는 왜 조정해야 해?
- similarity_threshold가 너무 높으면 어떻게 돼?
- MMR은 어떤 상황에서 유용해?
- RAG에서 temperature를 낮게 쓰는 이유는?
- 검색 실패와 생성 실패는 어떻게 달라?
- LLMOps 대시보드에서 어떤 지표를 봐야 해?

## 2. 설치 방법

```bash
cd rag_core_chatbot_for_team
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## 3. API 키 없이 실행하기

기본값은 **API 키 없는 실습 모드**입니다.

이 모드는 실제 LLM 생성이 아니라, 검색된 문서에서 관련 문장을 추출해 답변 형태로 보여줍니다. 팀원이 RAG의 retrieval, chunking, prompt, logging을 이해하는 데 초점을 둔 모드입니다.

## 4. OpenAI 생성 모드 사용하기

OpenAI 생성 모드를 쓰면 검색 문서를 프롬프트에 넣고 LLM이 답변을 생성합니다.

1. `.streamlit/secrets.example.toml`을 참고해 `.streamlit/secrets.toml`을 만듭니다.
2. 아래처럼 API 키를 넣습니다.

```toml
OPENAI_API_KEY="sk-your-api-key"
```

3. 앱 왼쪽 사이드바에서 `OpenAI 생성 모드`를 선택합니다.

주의: API 키는 GitHub에 올리면 안 됩니다.

## 5. 고급 임베딩 검색 사용하기

기본 검색은 TF-IDF입니다. 의미 기반 검색을 실습하려면 Sentence-Transformers를 사용할 수 있습니다.

```bash
pip install -r requirements-advanced.txt
streamlit run app.py
```

앱 사이드바에서 `Sentence-Transformers 임베딩 검색(고급)`을 선택합니다.

기본 모델:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

처음 실행 시 모델 다운로드가 필요하므로 인터넷 연결이 필요합니다.

## 6. 폴더 구조

```text
rag_core_chatbot_for_team/
├── app.py
├── README.md
├── requirements.txt
├── requirements-advanced.txt
├── src/
│   └── rag_pipeline.py
├── data/
│   ├── knowledge_base/
│   │   ├── 01_RAG_개요.md
│   │   ├── 02_RAG_파이프라인.md
│   │   └── ...
│   ├── logs/
│   │   ├── rag_run_logs.csv
│   │   └── retrieval_logs.csv
│   └── sample_questions.csv
├── docs/
│   ├── PARAMETER_CHEATSHEET.md
│   └── TEAM_ASSIGNMENT_GUIDE.md
└── .streamlit/
    ├── config.toml
    └── secrets.example.toml
```

## 7. 저장되는 로그

앱을 실행하면 `data/logs` 폴더에 두 종류의 로그가 저장됩니다.

### rag_run_logs.csv

질문 단위 실행 로그입니다.

- run_id
- question
- answer
- latency_ms
- retrieval_latency_ms
- generation_latency_ms
- retriever_backend
- retrieval_mode
- top_k
- chunk_size
- chunk_overlap
- similarity_threshold
- model_mode
- model_name
- temperature
- prompt_tokens_est
- completion_tokens_est
- total_tokens_est

### retrieval_logs.csv

검색 결과 단위 로그입니다.

- run_id
- question
- rank
- chunk_id
- doc_id
- title
- chunk_index
- score
- retriever_backend
- retrieval_mode

이 로그는 최종 RAG Quality Monitor 프로젝트의 LLMOps 대시보드 구조를 이해하기 위한 기초 데이터입니다.

## 8. 팀원 과제

각 팀원은 다음을 수행합니다.

1. 앱을 실행한다.
2. 예시 질문 3개 이상을 입력한다.
3. top_k를 3과 5로 바꿔 결과를 비교한다.
4. chunk_size를 400과 900으로 바꿔 결과를 비교한다.
5. similarity_threshold를 높였을 때 검색 결과가 어떻게 달라지는지 확인한다.
6. 실행 로그 CSV를 열어 어떤 값이 저장되는지 확인한다.
7. RAG 파이프라인을 본인 말로 5문장 정리한다.

## 9. 배포 방법

가장 쉬운 배포는 Streamlit Community Cloud입니다.

1. 이 폴더를 GitHub 저장소에 올립니다.
2. Streamlit Community Cloud에서 저장소를 연결합니다.
3. main file path를 `app.py`로 지정합니다.
4. OpenAI 생성 모드를 사용할 경우 Secrets에 `OPENAI_API_KEY`를 등록합니다.
5. 배포 후 URL을 팀원에게 공유합니다.

## 10. 확장 아이디어

이 실습 앱을 최종 프로젝트로 확장하려면 다음을 추가합니다.

- KorQuAD/KLUE-MRC/AI Hub 데이터셋 연결
- Chroma 또는 FAISS 벡터 DB 적용
- RAGAS 또는 TruLens 평가 점수 생성
- retrieval hit rate, faithfulness, hallucination rate 계산
- PyTorch 기반 RAG 실패 예측 모델 추가
- Streamlit 대시보드 고도화
- FastAPI 백엔드 분리

## v2 수정 사항: 질문별 답변 다양화

초기 버전의 API 키 없는 실습 모드는 단순히 상위 검색 문서의 문장을 추출했기 때문에, 여러 질문에서 답변이 비슷하게 보일 수 있었습니다. v2에서는 다음을 개선했습니다.

- 질문 의도 분류 추가: RAG 개요, 데이터, chunking, top_k, threshold, MMR, generation parameter, LLMOps 등으로 분류
- 질문별 답변 템플릿 추가: 질문 의도에 따라 핵심 답변과 파라미터 조정 관점이 다르게 출력
- 검색어 확장 추가: TF-IDF 검색이 의미를 놓치지 않도록 RAG 관련 동의어와 핵심어를 자동 추가
- 근거 문장 재랭킹 개선: 질문 키워드와 의도별 핵심어가 포함된 문장을 우선 선택

따라서 API 키 없이도 질문별로 서로 다른 구조의 답변을 확인할 수 있습니다. 다만 이 모드는 여전히 실제 LLM 생성이 아니라 교육용 추출/템플릿 방식입니다. 실제 생성형 답변을 보려면 OpenAI 생성 모드를 사용해야 합니다.

---

## v3 업데이트: RAG 논문 지식베이스 추가

이번 버전에는 RAG 관련 주요 논문 요약 노트가 추가되었습니다.

추가된 흐름은 다음과 같습니다.

- 검색 기반 언어모델: REALM, DPR, RAG, FiD, KILT
- 검색 품질 개선: GAR, HyDE, RAG-Fusion
- 생성 중 검색 제어: FLARE, Self-RAG, CRAG
- 긴 문서/복잡한 질의: Lost in the Middle, RAPTOR, GraphRAG
- 평가/운영: RAGAS, ARES, RAG Survey

추천 질문은 `data/paper_questions.csv`와 `docs/RAG_PAPER_READING_GUIDE.md`를 참고하세요.

주의: 논문 PDF 전체가 아니라 한국어 요약 노트가 들어 있습니다. 원문 전체를 넣으면 앱이 느려지고, 교육용 질문에는 구조화된 요약이 더 유용하기 때문입니다.

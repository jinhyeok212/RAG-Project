# Retrieval 결과로 답변을 생성하는 최소 RAG 데모

## Retrieval-only와 RAG의 차이

기존 Retrieval-only baseline은 사용자 질문과 가까운 기존 질문을 Chroma에서 찾아 보여준다. 이번 데모는 그 결과의 질문, 답변, 인텐트를 LLM 프롬프트에 넣어 하나의 한국어 답변을 생성한다.

전체 흐름은 다음과 같다.

```text
사용자 질문 → 기존 임베딩 모델 → Chroma Top-K 검색
           → 근거 제한 프롬프트 → Ollama/OpenAI → 최종 답변
```

- Retrieval은 관련된 기존 상담 자료를 찾는다. Chroma는 질문 벡터를 저장하고 cosine **distance**가 낮은 순서로 자료를 반환한다. distance는 similarity가 아니다.
- Generation은 찾은 자료를 읽기 쉬운 최종 답변으로 구성한다. LLM은 사실 저장소가 아니며, 검색 자료 밖의 정보를 보충하도록 맡기지 않는다.

검색이 틀리거나 모호하면 LLM에 잘못된 근거가 전달되므로 생성 답변도 틀릴 수 있다. 그래서 프롬프트는 자료에 없는 내용을 추측하지 않고, 충돌하면 하나를 고르지 않으며, 필요한 맥락을 사용자에게 다시 묻게 한다. 같은 질문에 서로 다른 가격이나 수량의 답변이 있는 데이터에서는 특히 중요하다.

프롬프트 지시만 믿지 않고, 가격·수량·날짜/시간을 묻는 질문에서 검색 답변에 서로 다른 명시적 값이 발견되면 LLM 호출 전에 충돌 안내를 반환한다. `이거`, `그거`, `저거`처럼 대상이 불명확한 짧은 질문도 상품명이나 매장 정보를 추가로 요청한다. 이 규칙은 보수적인 문자열 후보 탐지이므로 모든 의미 충돌을 발견하거나 완벽히 판정하는 장치는 아니다.

## 설정

`.env.example`을 참고하여 프로젝트 루트의 `.env`를 설정한다. 실제 API 키는 문서, 소스 코드 또는 로그에 넣지 않는다.

```env
LLM_PROVIDER=ollama

OLLAMA_MODEL=qwen2.5:3b
OLLAMA_BASE_URL=http://localhost:11434

OPENAI_API_KEY=
OPENAI_MODEL=

RAG_COLLECTION=super_questions
RAG_TOP_K=3
RAG_MAX_DISTANCE=
RAG_DEBUG_PROMPT=false
RAG_DRY_RUN=false
```

`RAG_MAX_DISTANCE`가 비어 있으면 거리 필터를 사용하지 않는다. 숫자를 설정하면 그 값 이하의 문서만 사용한다. 적절한 임계값은 데이터와 질문 분포를 이용한 별도 실험으로 정해야 하며, 이 데모는 특정 값을 권장하거나 성능을 주장하지 않는다.

`RAG_DEBUG_PROMPT=true`이면 LLM 호출 직전에 완성된 프롬프트를 출력한다. 프롬프트에는 사용자 질문과 각 검색 문서의 기존 질문, 기존 답변, 인텐트, 문서 번호가 들어간다.

## 실행 방법(Windows PowerShell)

프로젝트 의존성을 먼저 설치한다.

```powershell
C:\Python312\python.exe -m pip install -r requirements.txt
```

Ollama는 로컬에서 실행되므로 API 키가 필요 없고 데이터가 로컬 모델로 전달된다. 모델 파일과 실행 환경은 사용자가 관리해야 한다.

```powershell
ollama pull qwen2.5:3b
ollama serve
C:\Python312\python.exe scripts\rag_chat_demo.py
```

OpenAI는 원격 API를 사용하므로 네트워크, 유효한 `OPENAI_API_KEY`, 명시적인 `OPENAI_MODEL`이 필요하며 API 사용 비용이 발생할 수 있다. `.env`에서 세 값을 설정한 뒤 실행한다.

```powershell
C:\Python312\python.exe scripts\rag_chat_demo.py
```

API 비용 없이 검색과 프롬프트를 확인하려면 다음과 같이 설정한다.

```env
RAG_DRY_RUN=true
RAG_DEBUG_PROMPT=true
```

```powershell
C:\Python312\python.exe scripts\rag_chat_demo.py
```

DRY RUN은 임베딩, Chroma 검색, 프롬프트 작성, 검색 근거 출력까지만 수행하고 LLM을 호출하지 않는다. `exit`, `quit`, `종료` 중 하나를 입력하면 대화형 프로그램이 끝난다.

## 로그

질문별 결과는 `results/rag_generation/rag_chat_log.jsonl`에 JSON 한 줄씩 누적된다. 각 레코드에는 다음 값이 있다.

- `timestamp`, `user_query`
- `llm_provider`, `llm_model`
- `retrieved_documents`, `final_answer`, `error`
- `query_embedding_time_ms`, `retrieval_time_ms`, `generation_time_ms`, `total_time_ms`

API 키와 완성된 프롬프트는 로그에 저장하지 않는다. 로그 저장이 실패해도 상담 결과는 콘솔에 표시하며 별도 경고를 출력한다.

## 오류 및 근거 부족 처리

컬렉션 부재와 모델 로딩 실패는 시작 단계에서 한국어 안내로 표시한다. 검색 결과가 없거나 설정한 거리 조건을 만족하는 문서가 없으면 LLM을 호출하지 않고 질문을 구체화해 달라고 요청한다. 모든 검색 문서의 답변이 비어 있어도 호출을 생략한다. Ollama 연결 실패, OpenAI 키·모델 누락, 시간 초과, 빈 응답도 전체 traceback 대신 이해하기 쉬운 메시지로 반환하고 로그에 남긴다.

## 현재 한계와 개선 방향

- 검색 성능이 완벽하지 않음
- 인텐트가 같아도 실제 답변이 다를 수 있음
- 상품명과 이전 대화 맥락이 부족할 수 있음
- LLM이 검색 근거를 잘못 해석할 수 있음
- 생성 답변 품질 평가는 아직 수행하지 않음

또한 현재는 단일 질문 대화이며 이전 턴을 기억하지 않는다. 충돌 판단은 프롬프트 지시에 의존하고, 검색 거리 임계값도 검증되지 않았다. 이후에는 대표 질문으로 retrieval 및 생성 평가 세트를 만들고, 메타데이터 필터, 대화 맥락, 구조화된 충돌 감지, 인용 검증, 사람 평가를 단계적으로 추가할 수 있다.

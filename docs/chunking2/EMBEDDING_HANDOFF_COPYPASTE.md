# 임베딩 담당자 전달용 메시지

아래 내용을 그대로 복사해서 임베딩 담당자에게 전달하면 됩니다.

```text
안녕하세요. 청킹 산출물 전달드립니다.

작업 순서는 파싱 -> 청킹 -> 임베딩 순서입니다.

- 파싱: 원본 PDF/TXT/JSON 같은 자료를 공통 JSON Document 구조로 정리하는 단계
- 청킹: 파싱된 Document를 검색에 잘 걸리도록 작은 단위로 나누는 단계
- 임베딩: 청킹된 `text`를 벡터로 바꿔 Chroma 같은 DB에 넣는 단계

이번 작업에서는 원본 데이터를 컴퓨터가 읽기 좋은 공통 JSON Document 구조로 정리한 파싱 결과물을 확인했고, 그 결과를 바탕으로 청킹까지 완료했습니다.

여기서 파싱은 PDF/TXT/JSON 같은 원본 내용을 `document_id`, `title`, `retrieval_text`, `metadata`, `raw_record` 같은 공통 구조로 바꾸는 작업을 의미합니다.

Data Engineer 쪽 역할은 원본 PDF/TXT/JSON 파일을 읽고, 텍스트를 추출한 뒤 공통 Document 구조로 변환하는 범위로 보면 됩니다.

저희 쪽에서는 그 공통 Document 구조가 청킹과 임베딩에 필요한 입력 스키마를 만족하는지 검증했고, 그 결과를 기반으로 청킹 산출물을 만들었습니다.

전체 흐름은 아래와 같습니다.

PDF/TXT/JSON 원본
-> Loader
-> 텍스트 추출
-> 공통 Document 구조
-> Chunking
-> Embedding 담당자에게 전달

현재 임베딩 담당자분은 원본 문서, 파싱, Loader를 다시 다룰 필요 없이, 청킹이 완료된 Chroma-ready JSONL 파일을 사용하시면 됩니다.

임베딩 대상 파일은 각 실험 폴더 안의 *_chunks_chroma_ready.jsonl 파일입니다.

폴더 구조는 아래와 같습니다.

05.chunks/
  c2_section_800/
  exp_02_section_800_parent/
  exp_03_field_atomic_parent/

각 파일의 기본 구조는 다음과 같습니다.

{
  "id": "청크 고유 ID",
  "text": "임베딩할 텍스트",
  "metadata": {
    "document_id": "부모 문서 ID",
    "title": "정책명",
    "url": "원문 URL",
    "section": "청크 유형",
    "chunk_index": 0
  }
}

임베딩 시 사용 기준은 다음과 같습니다.

- 임베딩 대상 필드: text
- 벡터 DB 고유 ID: id
- 부모 문서 연결 키: metadata.document_id
- 검색 후처리 키: metadata.document_id, metadata.section, metadata.chunk_index
- 결과 표시용 정보: metadata.title, metadata.url, metadata.section, text

Data Engineer와 사전에 합의되어야 하는 주요 항목은 다음입니다.

- document_id 생성 규칙
- chunk_id 생성 규칙
- 문서 중복 제거 기준
- 청크 텍스트 필드명
- 원본 파일 형식 기록 방식
- 문서 버전 메타데이터
- 정답 문서 및 정답 청크 표시 방식
- 청크의 원문 위치 정보

현재 산출물에서는 document_id와 chunk_id가 이미 생성되어 있고, 임베딩용 파일에서는 id가 chunk_id 역할을 합니다.

Chroma 적재 시 문제가 될 수 있는 list/dict 형태의 metadata는 청킹 단계에서 이미 정리했습니다.
즉 metadata는 Chroma에 넣을 수 있도록 flat scalar 타입으로 변환되어 있습니다.

실험 후보는 3개입니다.

1. C2 기본형
- 위치: 05.chunks/c2_section_800/c2_section_800_chunks_chroma_ready.jsonl
- C2는 문서를 무작정 글자 수로 자른 게 아니라, 지원내용/자격조건/신청방법 같은 섹션 기준으로 먼저 나눈 방식입니다.
- 그리고 한 섹션이 너무 길 때만 검색하기 좋게 800자 이하로 다시 나눴습니다.
- 청크 수는 2,875개입니다.

2. EXP 02: C2 + Parent 확장형
- 위치: 05.chunks/exp_02_section_800_parent/exp_02_section_800_parent_chunks_chroma_ready.jsonl
- parent 문서: 05.chunks/exp_02_section_800_parent/exp_02_section_800_parent_parent_documents.jsonl
- 쉽게 말하면 작은 조각으로 찾고, 전체 문서로 답하는 방식입니다.
- 검색은 청크 단위로 하고, 답변 생성 시에는 metadata.document_id를 이용해 같은 정책의 parent document를 함께 가져올 수 있습니다.
- 청크 수는 2,875개입니다.

3. EXP 03: Field Atomic + Parent 확장형
- 위치: 05.chunks/exp_03_field_atomic_parent/exp_03_field_atomic_parent_chunks_chroma_ready.jsonl
- parent 문서: 05.chunks/exp_03_field_atomic_parent/exp_03_field_atomic_parent_parent_documents.jsonl
- C2가 지원내용/자격조건/신청방법 같은 큰 섹션 기준으로 나눈 방식이라면, EXP 03은 그 안의 세부 필드까지 더 잘게 나눈 방식입니다.
- 예를 들어 신청기간, 신청방법, 신청 URL, 최소 나이, 최대 나이, 소득 조건, 제출서류, 참고 URL 같은 값을 각각 따로 청크로 만들었습니다.
- 필드 단위 질문에 강한지 보기 위한 후보입니다.
- 청크 수는 8,084개입니다.

우선 임베딩 후에는 document-level qrels 기준으로 검색 성능을 비교하면 됩니다.

평가 파일은 아래에 있습니다.

02.eval_question_qrels/eval_questions.jsonl
02.eval_question_qrels/qrels.jsonl

먼저 볼 지표는 다음입니다.

- recall@1
- recall@3
- recall@5
- MRR@10

각 지표의 의미는 아래와 같습니다.

- recall@1: 검색 결과 1개만 봤을 때 정답 문서가 포함되어 있는 비율입니다. 사용자가 첫 번째 결과만 봐도 맞는지를 보는 지표입니다.
- recall@3: 검색 결과 상위 3개 안에 정답 문서가 포함되어 있는 비율입니다. RAG에서 top-3를 컨텍스트로 넣을 때 정답이 들어오는지를 봅니다.
- recall@5: 검색 결과 상위 5개 안에 정답 문서가 포함되어 있는 비율입니다. 검색 후보를 조금 넓혔을 때 정답을 놓치지 않는지 확인합니다.
- MRR@10: 상위 10개 결과 안에서 정답 문서가 얼마나 앞쪽에 나오는지를 보는 지표입니다. 정답이 1등이면 1점, 2등이면 0.5점, 3등이면 0.333점처럼 계산합니다.

쉽게 말하면 recall@k는 "정답이 top-k 안에 들어왔는가"를 보고, MRR@10은 "정답이 얼마나 높은 순위에 왔는가"를 봅니다.

전체 평균뿐 아니라 질문 유형별 성능도 같이 보면 좋습니다.

질문 유형은 특정 정책 찾기, 자격조건, 신청기간, 신청방법, 지원내용, 제출서류로 나뉩니다.

정리하면, 임베딩 담당자분은 각 실험의 *_chunks_chroma_ready.jsonl 파일에서 text를 임베딩하고, id를 벡터 ID로 저장하면 됩니다.
metadata.document_id는 parent 문서 연결과 검색 결과 묶기에 사용하면 됩니다.
```

## 더 짧은 버전

```text
청킹 산출물 전달드립니다.

임베딩에는 각 실험 폴더의 *_chunks_chroma_ready.jsonl 파일을 사용하시면 됩니다.
text를 임베딩하고, id를 벡터 DB 고유 ID로 쓰면 됩니다.
metadata.document_id는 원문 정책 문서 연결 키입니다.

원본 데이터는 파싱을 거쳐 공통 JSON Document 구조로 정리되어 있고,
저희 쪽에서는 그 Document를 검증한 뒤 청킹해서 임베딩용 파일로 변환했습니다.

실험은 3개입니다.

1. c2_section_800: 섹션 기반 기본 청킹, 2,875 chunks
2. exp_02_section_800_parent: C2 + parent document 확장, 2,875 chunks
3. exp_03_field_atomic_parent: C2보다 더 잘게 세부 필드 단위로 나눈 atomic chunk + parent 확장, 8,084 chunks

Chroma metadata 타입 이슈는 청킹 단계에서 해결했고, list/dict는 flat scalar 형태로 정리되어 있습니다.

평가는 qrels 기준으로 recall@1/3/5, MRR@10을 먼저 보면 됩니다.
recall@k는 정답 문서가 상위 k개 검색 결과 안에 들어왔는지 보는 지표이고, MRR@10은 정답 문서가 상위 10개 안에서 얼마나 앞 순위에 나왔는지 보는 지표입니다.
```

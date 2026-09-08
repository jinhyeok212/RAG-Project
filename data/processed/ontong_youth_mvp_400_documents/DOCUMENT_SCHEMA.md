# Ontong Youth RAG Document Schema v1

## 목적

온통청년 MVP 400개 정책을 RAG 파이프라인에 넣기 위한 공통 문서 구조다. 정책 1개가 문서 1개가 된다.

## 레코드 단위

`one policy = one document`

## 3번 Chunking 담당자가 사용할 필드

`retrieval_text`

`content`와 `raw_record`는 보존/검수용이며, 실제 청킹 대상은 `retrieval_text`다.

## 최상위 필드

- `document_id`: 문서 고유 ID. `ontong_youth_{plcyNo}`
- `title`: 정책명
- `source`: `ontong_youth`
- `url`: 대표 URL
- `content`: 주요 정책 필드를 라벨형 텍스트로 결합한 본문
- `retrieval_text`: 검색/청킹 대상 텍스트
- `metadata`: 분류, 기관, 기간, 조건, URL, 품질점수
- `raw_record`: 온통청년 API 원본 주요 필드

## 아직 하지 않은 작업

- 평가 질문 생성
- `ground_truth_document_ids` 지정
- Chunking
- Chunk ID 생성
- Embedding
- Chroma 저장

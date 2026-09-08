# 청킹 결과물 설명 문서

## 1. 이번 작업에서 한 일

이번 작업은 **파싱 결과 확인 -> 청킹 -> 임베딩 담당자 전달용 파일 생성**까지 진행한 작업입니다.

원본 정책 데이터 400개는 이미 공통 JSON Document 구조로 정리되어 있었고, 이번 작업에서는 그 파싱 결과가 청킹에 사용할 수 있는 형태인지 확인한 뒤 청킹을 진행했습니다.

쉽게 말하면, 원본 데이터를 바로 임베딩 담당자에게 넘긴 것이 아니라 다음 순서로 정리했습니다.

```text
원본 데이터
-> 파싱된 공통 Document 확인
-> 청킹
-> Chroma-ready JSONL 생성
-> 임베딩 담당자에게 전달
```

여기서 **파싱**은 원본 PDF/TXT/JSON/API 데이터처럼 형태가 제각각인 데이터를 `document_id`, `title`, `retrieval_text`, `metadata`, `raw_record` 같은 공통 구조로 바꾸는 작업입니다.

**청킹**은 파싱된 문서를 검색에 잘 걸리도록 작은 텍스트 단위로 나누는 작업입니다.

## 2. Git에 포함할 범위

이번에 git에 올릴 대상은 **청킹 결과물과 청킹 설명 문서만**입니다.

포함 대상:

```text
05.chunks/
```

이 폴더 안에는 실험별 청킹 결과물, Chroma 적재용 파일, 품질 리포트, 매니페스트, 설명 문서가 들어 있습니다.

이번 요청 기준으로 `scripts/`, `01.document/`, `02.eval_question_qrels/`, 루트의 회의용 md 파일은 git 반영 범위에서 제외해도 됩니다.

## 3. 폴더 구조

```text
05.chunks/
  README.md
  CHUNKING_RESULTS_GUIDE.md

  c2_section_800/
    c2_section_800_chunks.jsonl
    c2_section_800_chunks_chroma_ready.jsonl
    c2_section_800_manifest.json
    c2_section_800_quality_report.csv
    EXPERIMENT_BRIEF.md

  exp_02_section_800_parent/
    exp_02_section_800_parent_chunks.jsonl
    exp_02_section_800_parent_chunks_chroma_ready.jsonl
    exp_02_section_800_parent_parent_documents.jsonl
    exp_02_section_800_parent_manifest.json
    exp_02_section_800_parent_quality_report.csv
    EXPERIMENT_BRIEF.md

  exp_03_field_atomic_parent/
    exp_03_field_atomic_parent_chunks.jsonl
    exp_03_field_atomic_parent_chunks_chroma_ready.jsonl
    exp_03_field_atomic_parent_parent_documents.jsonl
    exp_03_field_atomic_parent_manifest.json
    exp_03_field_atomic_parent_quality_report.csv
    EXPERIMENT_BRIEF.md
```

## 4. 임베딩 담당자가 실제로 쓰면 되는 파일

임베딩 담당자는 각 실험 폴더 안의 `*_chunks_chroma_ready.jsonl` 파일을 사용하면 됩니다.

```text
05.chunks/c2_section_800/c2_section_800_chunks_chroma_ready.jsonl
05.chunks/exp_02_section_800_parent/exp_02_section_800_parent_chunks_chroma_ready.jsonl
05.chunks/exp_03_field_atomic_parent/exp_03_field_atomic_parent_chunks_chroma_ready.jsonl
```

파일 구조는 공통적으로 아래 형태입니다.

```json
{
  "id": "청크 고유 ID",
  "text": "임베딩할 텍스트",
  "metadata": {
    "document_id": "원본 정책 문서 ID",
    "title": "정책명",
    "url": "원문 URL",
    "section": "청크가 속한 섹션",
    "chunk_index": 0
  }
}
```

임베딩 기준:

- `text`: 실제로 임베딩할 텍스트
- `id`: 벡터 DB에 저장할 청크 고유 ID
- `metadata.document_id`: 같은 원본 정책 문서로 묶기 위한 키
- `metadata.title`: 검색 결과에서 보여줄 정책명
- `metadata.url`: 원문 확인용 URL
- `metadata.section`: 청크가 어떤 성격의 내용인지 나타내는 태그

Chroma는 metadata에 list나 dict 같은 복잡한 타입이 들어가면 적재 오류가 날 수 있습니다. 그래서 `*_chunks_chroma_ready.jsonl` 파일에서는 metadata를 Chroma에 바로 넣을 수 있도록 문자열, 숫자, boolean 같은 flat scalar 타입으로 정리해두었습니다.

## 5. 실험 1: C2 기본형

위치:

```text
05.chunks/c2_section_800/
```

임베딩용 파일:

```text
05.chunks/c2_section_800/c2_section_800_chunks_chroma_ready.jsonl
```

C2는 문서를 무작정 글자 수로 자른 게 아니라, 지원내용/자격조건/신청방법 같은 섹션 기준으로 먼저 나눈 방식입니다.

그리고 한 섹션이 너무 길 때만 검색하기 좋게 800자 이하로 다시 나눴습니다. 이때 앞뒤 문맥이 완전히 끊기지 않도록 120자 overlap을 적용했습니다.

예를 들어 하나의 정책 문서는 대략 이런 섹션으로 나뉩니다.

```text
overview: 정책 개요
support_content: 지원내용
eligibility: 자격조건
application: 신청기간, 신청방법, 신청 URL
screening: 심사방법
required_documents: 제출서류
notes: 기타사항
```

C2의 목적은 정책 문서의 구조를 유지하면서 검색에 불필요한 노이즈를 줄이는 것입니다. 정책 하나를 통째로 임베딩하면 너무 많은 내용이 한 벡터에 섞일 수 있고, 반대로 너무 잘게 자르면 문맥이 부족할 수 있습니다. C2는 그 중간 지점의 기본 후보입니다.

검증 결과:

```text
문서 수: 400개
청크 수: 2,875개
문서당 평균 청크 수: 7.188개
문서당 최소 청크 수: 5개
문서당 최대 청크 수: 12개
본문 청크 최대 길이: 800자
overlap: 120자
빈 청크: 0개
중복 chunk_id: 0개
Chroma metadata 타입 오류: 0개
```

## 6. 실험 2: EXP 02 C2 + Parent 확장형

위치:

```text
05.chunks/exp_02_section_800_parent/
```

임베딩용 파일:

```text
05.chunks/exp_02_section_800_parent/exp_02_section_800_parent_chunks_chroma_ready.jsonl
```

parent 문서 파일:

```text
05.chunks/exp_02_section_800_parent/exp_02_section_800_parent_parent_documents.jsonl
```

EXP 02는 **작은 조각으로 찾고, 필요하면 전체 문서로 답하는 방식**입니다.

청크 자체는 C2와 같습니다. 즉, 지원내용/자격조건/신청방법 같은 섹션 기준으로 먼저 나누고, 한 섹션이 너무 길 때만 800자 이하로 다시 나눴습니다.

C2와 다른 점은 parent document 파일이 같이 있다는 점입니다. 검색은 작은 청크 단위로 하되, 답변을 만들 때 검색된 청크의 `metadata.document_id`를 이용해서 같은 정책의 전체 문서를 다시 가져올 수 있습니다.

처리 흐름은 아래와 같습니다.

```text
1. chunks_chroma_ready.jsonl의 text를 임베딩한다.
2. 사용자 질문으로 vector search를 한다.
3. top-k 청크를 가져온다.
4. 검색된 청크의 metadata.document_id를 확인한다.
5. parent_documents.jsonl에서 같은 document_id를 가진 전체 정책 문서를 찾는다.
6. 답변 생성 시 검색 청크와 parent 문서를 함께 참고한다.
```

이 방식은 검색 정확도와 답변 안정성을 같이 보려는 후보입니다. 청크는 작아서 검색에는 유리하고, 답변할 때는 parent 문서로 문맥을 보강할 수 있습니다.

검증 결과:

```text
문서 수: 400개
청크 수: 2,875개
parent 문서 수: 400개
문서당 평균 청크 수: 7.188개
본문 청크 최대 길이: 800자
overlap: 120자
빈 청크: 0개
중복 chunk_id: 0개
누락 문서: 0개
Chroma metadata 타입 오류: 0개
```

## 7. 실험 3: EXP 03 Field Atomic + Parent 확장형

위치:

```text
05.chunks/exp_03_field_atomic_parent/
```

임베딩용 파일:

```text
05.chunks/exp_03_field_atomic_parent/exp_03_field_atomic_parent_chunks_chroma_ready.jsonl
```

parent 문서 파일:

```text
05.chunks/exp_03_field_atomic_parent/exp_03_field_atomic_parent_parent_documents.jsonl
```

EXP 03은 C2보다 더 잘게 나눈 방식입니다.

C2가 지원내용/자격조건/신청방법/제출서류 같은 큰 섹션 기준으로 나눈 방식이라면, EXP 03은 그 섹션 안의 세부 필드까지 더 잘게 나눕니다.

예를 들어 아래 같은 값을 각각 따로 청크로 만듭니다.

```text
신청기간
신청방법
신청 URL
최소 나이
최대 나이
소득 조건
제출서류
참고 URL
```

이 방식은 사용자의 질문이 "신청기간이 언제야?", "제출서류가 뭐야?", "나이 조건이 어떻게 돼?"처럼 답변 위치가 분명할 때 성능이 좋아지는지 확인하기 위한 후보입니다.

다만 너무 잘게 나누면 청크 하나만으로는 정책 전체 맥락이 부족할 수 있습니다. 그래서 EXP 03도 parent document 확장을 같이 제공합니다. 검색은 세부 필드 청크로 하고, 답변할 때는 `metadata.parent_document_id` 또는 `metadata.document_id`로 전체 정책 문서를 다시 가져올 수 있습니다.

검증 결과:

```text
문서 수: 400개
청크 수: 8,084개
parent 문서 수: 400개
문서당 평균 청크 수: 20.21개
본문 청크 최대 길이: 800자
overlap: 120자
빈 청크: 0개
중복 chunk_id: 0개
누락 문서: 0개
Chroma metadata 타입 오류: 0개
```

## 8. 각 파일의 역할

`*_chunks_chroma_ready.jsonl`

임베딩 담당자가 바로 사용하면 되는 파일입니다. `text`를 임베딩하고, `id`를 벡터 DB의 고유 ID로 저장하면 됩니다.

`*_chunks.jsonl`

청킹 결과의 원본형 파일입니다. Chroma-ready 파일보다 metadata 구조가 원래 형태에 가깝습니다. 분석이나 디버깅용으로 보면 됩니다.

`*_parent_documents.jsonl`

parent 확장형 실험에서 사용하는 원문 정책 문서 파일입니다. 검색된 청크의 `document_id`로 전체 정책 문서를 다시 찾을 때 사용합니다. C2 기본형에는 parent 문서 파일이 없습니다.

`*_manifest.json`

청킹 방식, 파일 경로, 청크 수, 길이 통계, 검증 결과를 기록한 파일입니다.

`*_quality_report.csv`

문서별 청크 수, 길이 등 품질 점검에 필요한 요약 리포트입니다.

`EXPERIMENT_BRIEF.md`

각 실험 폴더 안에 있는 간단 설명 문서입니다. 실험별 핵심만 빠르게 확인할 때 사용합니다.

## 9. 평가할 때 보면 좋은 지표

평가는 document-level qrels 기준으로 먼저 비교하면 됩니다.

추천 지표:

```text
recall@1
recall@3
recall@5
MRR@10
```

지표 의미:

- `recall@1`: 검색 결과 1개만 봤을 때 정답 문서가 들어있는 비율입니다.
- `recall@3`: 검색 결과 상위 3개 안에 정답 문서가 들어있는 비율입니다.
- `recall@5`: 검색 결과 상위 5개 안에 정답 문서가 들어있는 비율입니다.
- `MRR@10`: 정답 문서가 상위 10개 안에서 얼마나 앞 순위에 나오는지 보는 지표입니다.

쉽게 말하면 `recall@k`는 정답이 top-k 안에 들어왔는지를 보는 지표이고, `MRR@10`은 정답이 얼마나 높은 순위에 나왔는지를 보는 지표입니다.

## 10. 프로젝트 복기용 요약

이번 청킹은 정책 RAG 검색 성능을 비교하기 위해 세 가지 방식으로 나누었습니다.

첫 번째 C2는 기본형입니다. 정책 문서를 섹션 기준으로 나누고, 한 섹션이 너무 길 때만 800자 이하로 다시 나눴습니다. 검색 성능과 문맥 유지의 균형을 보려는 후보입니다.

두 번째 EXP 02는 C2에 parent document 확장을 붙인 방식입니다. 검색은 C2와 같은 작은 청크로 하고, 답변할 때는 원문 정책 전체 문서를 함께 가져올 수 있습니다.

세 번째 EXP 03은 C2보다 더 잘게, 세부 필드 단위까지 나눈 방식입니다. 신청기간, 신청방법, 제출서류, 나이조건처럼 위치가 명확한 질문에 강한지 보기 위한 후보입니다. 이 방식도 parent document 확장을 제공합니다.

임베딩 담당자는 각 실험 폴더의 `*_chunks_chroma_ready.jsonl` 파일만 사용하면 됩니다. `text`를 임베딩하고, `id`를 벡터 DB ID로 쓰고, `metadata.document_id`로 원본 정책 문서와 연결하면 됩니다.


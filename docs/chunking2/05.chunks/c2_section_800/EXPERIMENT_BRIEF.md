# C2 청킹 산출물 임베딩 인계 브리핑

## 한 줄 요약

정책 문서 400개를 C2 방식으로 섹션 기반 청킹했고, Chroma에 바로 넣을 수 있도록 metadata 타입 정리까지 완료했습니다.

## C2 방식이란?

C2는 이번 실험 후보 중 "섹션 기반 청킹 + 긴 섹션만 800자 이하로 추가 분할" 방식입니다.

정책 문서를 무작정 글자 수로 자른 게 아니라, 먼저 지원내용/자격조건/신청방법 같은 섹션 기준으로 나눕니다. 그리고 한 섹션이 너무 길 때만 검색하기 좋게 800자 이하로 다시 나눴습니다. 이때 앞뒤 문맥이 끊기지 않도록 120자 overlap을 둡니다.

이 방식의 목표는 정책 문서의 구조를 유지하면서도, 검색할 때 너무 긴 문서 전체가 한 번에 들어가서 노이즈가 커지는 문제를 줄이는 것입니다. 그래서 자격, 신청기간, 신청방법, 제출서류처럼 평가 질문 유형이 명확한 데이터에 균형이 좋습니다.

## 전달 파일

- 임베딩용 파일: `05.chunks/c2_section_800/c2_section_800_chunks_chroma_ready.jsonl`
- 매니페스트: `05.chunks/c2_section_800/c2_section_800_manifest.json`
- 품질 리포트: `05.chunks/c2_section_800/c2_section_800_quality_report.csv`
- 원본형 청크 파일: `05.chunks/c2_section_800/c2_section_800_chunks.jsonl`

임베딩에는 `c2_section_800_chunks_chroma_ready.jsonl`을 사용하면 됩니다.

## 청킹 방식

- 방식: C2 section-based chunking
- 대상: `01.document/documents.jsonl`
- 원본 문서 수: 400개
- 청킹 기준: 정책 문서의 구조를 살려 섹션별로 분리
- 긴 섹션 처리: body 기준 800자 초과 시 분할
- overlap: 120자
- 헤더 포함: 각 청크 앞에 정책명, document_id, 분류, 키워드, 기관, 섹션 정보를 포함

섹션은 다음 기준으로 나눴습니다.

- `overview`: 정책명, 요약, 분류, 키워드, 기관
- `support_content`: 지원내용
- `eligibility`: 참여대상, 추가 자격, 나이/소득/혼인/지역 조건
- `application`: 신청기간, 사업기간, 신청방법, 신청 URL
- `screening`: 심사방법
- `required_documents`: 제출서류
- `notes`: 기타사항, 참고 URL

## 임베딩 파일 구조

`c2_section_800_chunks_chroma_ready.jsonl`은 한 줄에 청크 하나이며, 구조는 다음과 같습니다.

```json
{
  "id": "chunk_id",
  "text": "임베딩할 텍스트",
  "metadata": {
    "document_id": "...",
    "title": "...",
    "url": "...",
    "section": "...",
    "chunk_index": 0
  }
}
```

임베딩 담당자에게 요청할 기준은 아래와 같습니다.

- 임베딩 대상 필드: `text`
- 벡터 ID: `id`
- 부모 문서 ID: `metadata.document_id`
- 검색 후처리 키: `metadata.document_id`, `metadata.section`, `metadata.chunk_index`
- 결과 표시용 필드: `metadata.title`, `metadata.url`, `metadata.section`, `text`

## 검증 결과

- 생성 청크 수: 2,875개
- 문서당 평균 청크 수: 7.188개
- 문서당 최소/최대 청크 수: 5개 / 12개
- 빈 청크: 0개
- 중복 chunk_id: 0개
- 누락 문서: 0개
- body_text 최대 길이: 800자
- 헤더 포함 text 최대 길이: 1,017자
- Chroma metadata 타입 오류: 0개

## Chroma-ready 처리

Chroma는 metadata에 list/dict 같은 복합 타입이 들어가면 문제가 날 수 있어서, 청킹 단계에서 임베딩용 파일을 별도로 만들었습니다.

`normalized_keywords`, `normalized_lclsf_list`, `normalized_mclsf_list` 같은 리스트 필드는 문자열로 변환했습니다. 따라서 임베딩 담당자는 `c2_section_800_chunks_chroma_ready.jsonl`을 그대로 사용하면 됩니다.

## 미팅에서 말할 내용

이번 산출물은 최종 운영 확정본이라기보다는, 임베딩과 검색 성능 실험에 바로 넣을 수 있는 C2 후보입니다. 청킹 산출물 자체 검증은 완료됐고, 다음 단계에서는 문서 단위 qrels 기준으로 `recall@1/3/5`, `MRR@10`을 먼저 보면 됩니다.

검색 결과는 chunk 단위로 뽑되, `document_id`가 보존되어 있으므로 이후 같은 문서의 청크를 묶거나 parent document를 확장하는 후처리 실험도 가능합니다.

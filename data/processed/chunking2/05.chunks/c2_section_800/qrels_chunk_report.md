# C2 Chunk Qrels Final Report

- created_at: 2026-09-14T13:43:49.401515+00:00
- questions: 47
- C2 chunks: 2875
- C2 documents: 400
- qrels rows: 50
- mapped questions: 47 / 47
- unmapped questions: 0
- multi-chunk questions: 3

## Mapping Rules

- plcySprtCn: support_content
- addAplyQlfcCndCn+ptcpPrpTrgtCn: eligibility
- aplyYmd: application
- plcyAplyMthdCn: application
- sbmsnDcmntCn: required_documents
- retrieval_text: overview

위 규칙으로 먼저 후보 Chunk를 정한 뒤, 47개 질문의 질문·Reference Answer·Gold Document·Chunk 본문을 직접 대조했다. 필드 저장 위치와 실제 답 근거가 다른 경우에는 실제 답을 포함한 Chunk를 Gold로 선택했다.

## Validation

- unique chunk_id count: 2875
- duplicate chunk_id count: 0
- blank chunk_id count: 0
- blank document_id count: 0
- blank text count: 0
- Document qrels Gold pair coverage: 50 / 50
- semantic evidence review: 50 / 50
- duplicate qrels row count: 0
- unknown chunk_id count: 0

## Added Gold Chunks

- `oy_eval_repair_q0037`: 복수 Gold Document의 `support_content` Chunk 1개 추가
- `oy_eval_repair_q0038`: 복수 Gold Document의 `eligibility` Chunk 1개 추가
- `oy_eval_repair_q0080`: 복수 Gold Document의 `application` Chunk 1개 추가

## Removed Non-Evidence Chunks

- `oy_eval_repair_q0038`: 지역 코드만 포함하고 질문의 정답 근거를 포함하지 않는 `eligibility__001~003` Chunk 3개 제외

## Semantic Review Corrections

- `oy_eval_repair_q0066`: 요약된 자격값만 있는 `eligibility__000` 대신, 연령·거주·소득·혼인·주택·제외 조건을 직접 설명하는 `support_content__000`으로 교체
- `oy_eval_repair_q0073`: 우선선발 조건만 있는 `eligibility__000` 대신, 서울 거주·연령·제대군인 연령 연장·코스별 연령을 직접 설명하는 `support_content__000`으로 교체

## Notes

- 47개 평가 질문에 사용하는 검수 완료 Chunk 단위 Gold 파일이다.
- `retrieval_text` 질문은 해당 정책 문서를 찾는 질문이므로 정책명과 요약을 포함한 `overview` Chunk를 Gold로 선택했다.
- 한 질문에 Gold Document가 여러 개인 경우에는 각 Gold Document에서 답 근거가 되는 최소 1개 Chunk를 포함했다.
- 긴 필드가 여러 Chunk로 분할되어도 지역 코드처럼 질문의 답을 설명하지 않는 조각은 Gold에서 제외했다.

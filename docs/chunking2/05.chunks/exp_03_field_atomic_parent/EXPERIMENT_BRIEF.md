# EXP 03 Field Atomic Parent 청킹 설명

## 한 줄 요약

C2보다 더 잘게, 정책 문서의 세부 필드 단위까지 나누고 parent document 확장용 파일을 함께 제공하는 후보입니다.

## 전달 파일

- 임베딩용 파일: `05.chunks\exp_03_field_atomic_parent\exp_03_field_atomic_parent_chunks_chroma_ready.jsonl`
- parent 확장용 문서 파일: `05.chunks\exp_03_field_atomic_parent\exp_03_field_atomic_parent_parent_documents.jsonl`
- 매니페스트: `05.chunks\exp_03_field_atomic_parent\exp_03_field_atomic_parent_manifest.json`
- 품질 리포트: `05.chunks\exp_03_field_atomic_parent\exp_03_field_atomic_parent_quality_report.csv`
- 원본형 청크: `05.chunks\exp_03_field_atomic_parent\exp_03_field_atomic_parent_chunks.jsonl`

## 방식 설명

- C2가 지원내용, 자격조건, 신청방법, 제출서류 같은 큰 섹션 기준으로 나눈 방식이라면, EXP 03은 그 섹션 안의 세부 필드까지 더 잘게 나눈 방식입니다.
- 예를 들어 신청기간, 신청방법, 신청 URL, 최소 나이, 최대 나이, 소득 조건, 제출서류, 참고 URL 같은 값을 각각 atomic chunk로 만듭니다.
- 각 청크에는 `section`과 `field` 태그가 같이 들어갑니다.
- 신청기간, 신청방법, 제출서류, 자격조건처럼 답변 위치가 명확한 질문에 강한지 확인하기 위한 후보입니다.
- 검색 결과가 너무 세부적일 수 있으므로 `metadata.parent_document_id`로 전체 정책 문서를 확장하는 전제를 둡니다.

## 검증 결과

- 문서 수: 400
- 청크 수: 8084
- 문서당 평균 청크 수: 20.21
- 빈 청크: 0
- 중복 chunk_id: 0
- 누락 문서: 0
- Chroma metadata 타입 오류: 0

# EXP 02 Section 800 Parent 청킹 설명

## 한 줄 요약

C2 섹션 기반 청킹 결과에 parent document 확장용 파일을 함께 제공하는 후보입니다.

## 전달 파일

- 임베딩용 파일: `05.chunks\exp_02_section_800_parent\exp_02_section_800_parent_chunks_chroma_ready.jsonl`
- parent 확장용 문서 파일: `05.chunks\exp_02_section_800_parent\exp_02_section_800_parent_parent_documents.jsonl`
- 매니페스트: `05.chunks\exp_02_section_800_parent\exp_02_section_800_parent_manifest.json`
- 품질 리포트: `05.chunks\exp_02_section_800_parent\exp_02_section_800_parent_quality_report.csv`
- 원본형 청크: `05.chunks\exp_02_section_800_parent\exp_02_section_800_parent_chunks.jsonl`

## 방식 설명

- C2와 같은 섹션 기반 청킹을 사용합니다.
- 청크 검색은 `text` 기준으로 수행하고, 검색 후 `metadata.parent_document_id`로 원문 정책 문서를 확장할 수 있게 했습니다.
- parent 확장용 전체 문서는 `parent_documents.jsonl`의 `text` 필드에 들어 있습니다.
- 목적은 세부 항목 검색 정확도와 답변 생성 시 전체 정책 맥락을 함께 가져가는 것입니다.

## 검증 결과

- 문서 수: 400
- 청크 수: 2875
- 문서당 평균 청크 수: 7.188
- 빈 청크: 0
- 중복 chunk_id: 0
- 누락 문서: 0
- Chroma metadata 타입 오류: 0

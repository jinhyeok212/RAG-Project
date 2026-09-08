# Chunking Experiments

이 폴더는 청킹 실험 산출물을 실험별로 분리해 둔 위치입니다.

전체 작업 배경, 실험별 차이, 임베딩 담당자 사용 방법은 `CHUNKING_RESULTS_GUIDE.md`에 정리되어 있습니다.

## 폴더 구성

- `c2_section_800`
  - C2 기본 후보
  - 섹션 기반 청킹
  - 한 섹션이 너무 길 때만 800자 이하로 추가 분할
  - overlap 120자
  - parent 확장 없음

- `exp_02_section_800_parent`
  - D1 후보
  - C2와 같은 청크 구성
  - 검색 후 parent document 확장 가능

- `exp_03_field_atomic_parent`
  - E1 후보
  - 필드 단위 atomic chunk
  - 검색 후 parent document 확장 가능

## 임베딩용 파일

각 실험 폴더 안의 `*_chunks_chroma_ready.jsonl` 파일을 임베딩에 사용하면 됩니다.

공통 규칙:

- 임베딩 대상 필드: `text`
- 벡터 ID: `id`
- 부모 문서 ID: `metadata.document_id`
- Chroma metadata: flat scalar 타입으로 정리 완료

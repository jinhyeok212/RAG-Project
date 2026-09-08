# Ontong Youth Evaluation Question Candidate Schema

## 목적

MVP 400개 정책 문서에서 retrieval 평가용 질문 후보를 생성한다. 이 파일은 최종 평가셋이 아니며, 사람이 검수해야 한다.

## 주요 파일

- `eval_question_candidates_review.csv`: 사람이 검수할 CSV
- `eval_question_candidates.jsonl`: 후보 질문 JSONL
- `qrels_candidates.jsonl`: 후보 질문과 정답 문서 연결

## 중요 필드

- `query_id`: 질문 ID
- `query`: 사용자 질문 형태의 평가 질문
- `ground_truth_document_ids`: 정답 문서 ID 배열
- `question_type`: 질문 유형
- `reference_answer`: 정답 문서에서 가져온 참고 답변. LLM 평가 정답이 아니라 검수 참고용
- `human_review_label`: 사람이 입력할 검수 라벨

## 검수 라벨

- `1`: 평가 질문으로 사용
- `0`: 평가 질문에서 제외
- `2`: 수정/보류 필요

## 주의

아직 chunking, embedding, Chroma, retrieval 평가는 수행하지 않았다.

# 임베딩 담당자에게 설명할 내용

## 1. 우리가 지금 한 일

작업 순서는 **파싱 -> 청킹 -> 임베딩**입니다.

- **파싱**: 원본 PDF/TXT/JSON 같은 자료를 공통 JSON Document 구조로 정리하는 단계
- **청킹**: 파싱된 Document를 검색에 잘 걸리도록 작은 단위로 나누는 단계
- **임베딩**: 청킹된 `text`를 벡터로 바꿔 Chroma 같은 DB에 넣는 단계

우리는 원본 정책 문서 400개를 컴퓨터가 읽기 좋은 공통 JSON Document 구조로 정리한 파싱 결과물을 확인했고, 그 결과를 RAG 검색에 넣기 좋게 청킹했습니다.

여기서 파싱은 원본 문서 내용을 `document_id`, `title`, `retrieval_text`, `metadata`, `raw_record` 같은 구조로 바꾸는 작업입니다.

쉽게 말하면, 긴 정책 문서를 그대로 임베딩하지 않고 검색하기 좋은 작은 조각들로 나눴습니다.

이번에 만든 청킹 결과물은 임베딩 담당자가 바로 벡터 DB에 넣어볼 수 있는 상태입니다.

## 2. 임베딩 담당자가 하면 되는 일

임베딩 담당자는 원본 문서를 다시 파싱하거나 청킹할 필요가 없습니다.

우리가 넘기는 `*_chunks_chroma_ready.jsonl` 파일을 읽고, 그 안의 `text` 필드만 임베딩하면 됩니다.

기본 규칙은 이것입니다.

```text
임베딩할 값: text
벡터 DB 고유 ID: id
원문 정책 문서 연결 키: metadata.document_id
검색 결과 표시용 정보: metadata.title, metadata.url, metadata.section
```

## 3. 파일 구조

임베딩용 파일은 이런 형태입니다.

```json
{
  "id": "청크 고유 ID",
  "text": "임베딩할 텍스트",
  "metadata": {
    "document_id": "부모 정책 문서 ID",
    "title": "정책명",
    "url": "원문 URL",
    "section": "청크 내용 유형",
    "chunk_index": 0
  }
}
```

여기서 가장 중요한 것은 세 가지입니다.

- `id`: 벡터 DB에 저장할 때 쓰는 청크 고유 ID
- `text`: 실제로 임베딩해야 하는 텍스트
- `metadata.document_id`: 같은 정책 문서의 다른 청크나 원문으로 돌아갈 때 쓰는 ID

## 4. 청킹 실험은 3개로 나눴음

현재 `05.chunks` 폴더 안에 실험별로 폴더를 분리해뒀습니다.

```text
05.chunks/
  c2_section_800/
  exp_02_section_800_parent/
  exp_03_field_atomic_parent/
```

각 폴더 안에서 임베딩에 사용할 파일은 `*_chunks_chroma_ready.jsonl`입니다.

## 5. C2 기본형

위치:

```text
05.chunks/c2_section_800/c2_section_800_chunks_chroma_ready.jsonl
```

C2는 이번 청킹 실험의 기본 후보입니다.

C2는 문서를 무작정 글자 수로 자른 게 아니라, 지원내용/자격조건/신청방법 같은 섹션 기준으로 먼저 나눈 방식입니다.

예를 들면 다음과 같습니다.

- 정책 개요
- 지원내용
- 자격조건
- 신청기간/신청방법
- 심사방법
- 제출서류
- 기타사항

그리고 한 섹션이 너무 길 때만 검색하기 좋게 800자 이하로 다시 나눴습니다.

즉, C2는 **섹션 구조를 먼저 살리고, 너무 긴 섹션만 800자 이하 청크로 나눈 방식**입니다.

검증 결과:

```text
문서 수: 400개
청크 수: 2,875개
빈 청크: 0개
중복 ID: 0개
Chroma metadata 타입 오류: 0개
```

## 6. EXP 02: C2 + Parent 확장형

위치:

```text
05.chunks/exp_02_section_800_parent/exp_02_section_800_parent_chunks_chroma_ready.jsonl
```

parent 문서 파일:

```text
05.chunks/exp_02_section_800_parent/exp_02_section_800_parent_parent_documents.jsonl
```

EXP 02는 쉽게 말해 **작은 조각으로 찾고, 전체 문서로 답하는 방식**입니다.

사용자 질문이 들어오면 먼저 작게 나뉜 청크들 중에서 가장 관련 있는 청크를 찾습니다. 그다음 검색된 청크의 `document_id`를 보고, 같은 정책의 원문 전체 문서인 parent document를 다시 가져옵니다.

즉, 검색은 청크 단위로 정밀하게 하고 답변은 원문 정책 전체 맥락까지 참고해서 더 안정적으로 만드는 방식입니다.

기술적으로 말하면, EXP 02는 **C2 청킹 결과에 parent document 확장 기능을 붙인 실험**입니다.

청크 자체는 C2와 같습니다.

즉, 정책 문서를 다음 섹션으로 나눕니다.

- `overview`
- `support_content`
- `eligibility`
- `application`
- `screening`
- `required_documents`
- `notes`

그리고 한 섹션이 너무 길 때만 검색하기 좋게 800자 이하로 다시 나눕니다.

C2와 다른 점은, 검색된 청크에서 바로 답변을 끝내지 않고 필요하면 같은 `document_id`를 가진 원문 정책 문서 전체를 다시 가져올 수 있게 준비했다는 점입니다.

쉽게 말하면:

```text
검색은 작은 청크로 정확하게 하고,
답변할 때는 필요하면 원문 정책 문서 전체를 같이 참고하는 방식
```

예를 들어 사용자가 이렇게 물었다고 가정합니다.

```text
부산에서 중소기업 재직 청년이 복지포인트를 신청하려면 어떤 서류가 필요해?
```

이 질문은 먼저 `required_documents` 청크나 `eligibility` 청크가 검색될 가능성이 큽니다.

그런데 검색된 청크 하나만 보면 제출서류는 알 수 있어도, 이게 정확히 어떤 정책인지, 신청기간은 언제인지, 예외조건은 있는지까지는 부족할 수 있습니다.

그래서 EXP 02에서는 검색된 청크의 `metadata.document_id`를 사용해서 parent 문서 파일에서 같은 정책의 전체 문서를 다시 가져올 수 있게 했습니다.

처리 흐름은 다음과 같습니다.

```text
1. chunks_chroma_ready.jsonl의 text를 임베딩한다.
2. 사용자 질문으로 vector search를 한다.
3. top-k 청크를 가져온다.
4. 각 청크의 metadata.document_id를 확인한다.
5. parent_documents.jsonl에서 같은 document_id의 전체 정책 문서를 찾는다.
6. 답변 생성 시 검색 청크 + parent 문서를 함께 사용한다.
```

EXP 02에서 중요한 필드는 다음과 같습니다.

```text
id: 벡터 DB에 저장할 청크 ID
text: 임베딩할 청크 텍스트
metadata.document_id: parent 문서를 찾는 키
metadata.parent_document_id: parent 확장용 키, document_id와 같은 값
metadata.section: 청크가 어떤 섹션인지 알려주는 태그
metadata.chunk_index: 같은 문서 안에서 청크 순서를 복원할 때 쓰는 값
```

이 실험의 장점은 C2의 검색 정확도를 유지하면서 답변 생성 단계에서 문맥 부족 문제를 줄일 수 있다는 점입니다.

특히 이런 질문에서 유리할 수 있습니다.

- 특정 정책을 찾는 질문
- 신청방법과 자격조건을 같이 봐야 하는 질문
- 제출서류와 예외조건을 함께 확인해야 하는 질문
- 검색된 청크만으로 답변하기에는 맥락이 부족한 질문

주의할 점도 있습니다.

parent 문서를 항상 붙이면 답변 생성 컨텍스트가 길어질 수 있습니다. 그래서 처음에는 top-k 청크를 기준으로 찾되, 답변 생성 단계에서 필요한 경우에만 parent 문서를 확장하는 방식이 좋습니다.

검증 결과:

```text
문서 수: 400개
청크 수: 2,875개
parent 문서 수: 400개
빈 청크: 0개
중복 ID: 0개
누락 문서: 0개
Chroma metadata 타입 오류: 0개
```

회의에서는 이렇게 설명하면 됩니다.

```text
EXP 02는 C2와 같은 청크를 사용하지만,
검색 후 document_id로 원문 정책 문서 전체를 다시 가져올 수 있게 만든 버전입니다.

즉 검색은 청크 단위로 정밀하게 하고,
답변 생성에서는 필요할 때 parent document를 붙여서 문맥을 보강하는 방식입니다.

그래서 C2가 검색 성능을 보는 기본형이라면,
EXP 02는 실제 RAG 답변 품질까지 고려한 후보입니다.
```

## 7. EXP 03: 필드 단위 Atomic + Parent 확장형

위치:

```text
05.chunks/exp_03_field_atomic_parent/exp_03_field_atomic_parent_chunks_chroma_ready.jsonl
```

parent 문서 파일:

```text
05.chunks/exp_03_field_atomic_parent/exp_03_field_atomic_parent_parent_documents.jsonl
```

EXP 03은 C2보다 더 잘게 나눈 방식입니다.

C2는 정책 문서를 `지원내용`, `자격조건`, `신청방법`, `제출서류`처럼 큰 섹션 기준으로 나눴습니다.

EXP 03은 여기서 한 단계 더 들어가서, 섹션 안에 있는 세부 필드까지 따로 나눴습니다.

예를 들어 `신청기간`, `신청방법`, `신청 URL`, `최소 나이`, `최대 나이`, `소득 조건`, `제출서류`, `참고 URL` 같은 값을 각각 따로 청크로 만들었습니다.

즉, EXP 03은 **C2처럼 섹션 단위로만 나누는 게 아니라, 신청기간/신청방법/제출서류/나이조건 같은 세부 필드 단위까지 더 잘게 나눈 방식**입니다.

쉽게 말하면:

```text
질문이 "신청방법이 뭐야?", "제출서류가 뭐야?"처럼 명확할 때
해당 필드 청크를 바로 찾게 하려는 방식
```

대신 너무 잘게 나뉘어서 문맥이 부족할 수 있으므로, 이 실험도 parent 문서 확장을 같이 사용합니다.

검증 결과:

```text
문서 수: 400개
청크 수: 8,084개
빈 청크: 0개
중복 ID: 0개
Chroma metadata 타입 오류: 0개
```

## 8. Chroma-ready 처리는 끝났음

Chroma는 metadata에 리스트나 딕셔너리 같은 복잡한 값이 들어가면 적재 오류가 날 수 있습니다.

그래서 청킹 단계에서 이미 Chroma에 넣기 좋은 형태로 정리했습니다.

예를 들어 `normalized_keywords`처럼 원래 리스트였던 값은 문자열로 바꿨습니다.

따라서 임베딩 담당자는 `*_chunks_chroma_ready.jsonl` 파일을 그대로 사용하면 됩니다.

## 9. 평가할 때 봐야 하는 것

평가 파일은 여기에 있습니다.

```text
02.eval_question_qrels/eval_questions.jsonl
02.eval_question_qrels/qrels.jsonl
```

우선 문서 단위로 제대로 찾는지를 보면 됩니다.

볼 지표는 다음과 같습니다.

```text
recall@1
recall@3
recall@5
MRR@10
```

각 지표의 의미는 다음과 같습니다.

- `recall@1`: 검색 결과 1개만 봤을 때 정답 문서가 포함되어 있는 비율입니다. 첫 번째 결과가 바로 맞는지를 확인합니다.
- `recall@3`: 검색 결과 상위 3개 안에 정답 문서가 포함되어 있는 비율입니다. top-3를 RAG 컨텍스트로 넣을 때 정답 근거가 들어오는지 봅니다.
- `recall@5`: 검색 결과 상위 5개 안에 정답 문서가 포함되어 있는 비율입니다. 후보를 조금 넓혔을 때 정답을 놓치지 않는지 확인합니다.
- `MRR@10`: 상위 10개 결과 안에서 정답 문서가 얼마나 앞쪽에 나오는지를 보는 지표입니다. 정답이 1등이면 1점, 2등이면 0.5점, 3등이면 0.333점처럼 계산합니다.

쉽게 말하면 `recall@k`는 **정답이 top-k 안에 들어왔는가**를 보고, `MRR@10`은 **정답이 얼마나 높은 순위에 왔는가**를 봅니다.

그리고 전체 평균만 보지 말고 질문 유형별로 나눠서 봐야 합니다.

질문 유형은 다음과 같습니다.

- 특정 정책 찾기
- 자격조건
- 신청기간
- 신청방법
- 지원내용
- 제출서류

## 10. 회의에서 내가 말하면 되는 문장

아래처럼 말하면 됩니다.

```text
청킹 산출물은 실험별로 분리해뒀습니다.

임베딩 담당자는 원본 문서나 파싱 단계를 다시 만질 필요 없이,
각 실험 폴더 안의 *_chunks_chroma_ready.jsonl 파일을 사용하면 됩니다.

파일 구조는 id, text, metadata입니다.
text만 임베딩하면 되고, id는 벡터 DB의 고유 ID로 사용하면 됩니다.
metadata.document_id는 원문 정책 문서와 연결하는 parent key입니다.

청킹 실험은 세 가지입니다.

첫 번째 C2는 정책 문서를 섹션 단위로 나눈 기본형입니다.
지원내용, 자격조건, 신청방법, 제출서류 같은 의미 단위로 나눴고,
한 섹션이 너무 길 때만 검색하기 좋게 800자 이하로 다시 나눴습니다.

두 번째 EXP 02는 C2와 같은 청크를 쓰되,
검색 후 document_id로 원문 정책 문서를 확장할 수 있게 만든 버전입니다.

세 번째 EXP 03은 신청기간, 신청방법, 제출서류 같은 필드 단위로 더 잘게 나눈 버전입니다.
이것도 parent 문서 확장이 가능합니다.

Chroma metadata 타입 문제는 청킹 단계에서 해결했습니다.
리스트 타입은 문자열로 변환했고, 타입 검증도 통과했습니다.

평가는 qrels 기준으로 recall@1, recall@3, recall@5, MRR@10을 보면 됩니다.
전체 평균뿐 아니라 질문 유형별 성능도 같이 비교하면 됩니다.
```

## 11. 더 짧게 말해야 하면

시간이 없으면 이렇게만 말해도 됩니다.

```text
임베딩용 파일은 각 실험 폴더의 *_chunks_chroma_ready.jsonl입니다.
text 필드를 임베딩하고, id를 벡터 ID로 쓰면 됩니다.
metadata.document_id는 원문 정책 문서 연결 키입니다.

실험은 C2 기본형, C2+parent 확장형, field atomic+parent 확장형 세 가지입니다.
Chroma metadata 타입 정리는 청킹 단계에서 완료했습니다.
이후 qrels 기준으로 recall@1/3/5와 MRR@10을 비교하면 됩니다.
```

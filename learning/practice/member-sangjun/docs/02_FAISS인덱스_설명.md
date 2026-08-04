# 02_build_index.py 설명

이 단계는 문서를 청크로 나누고, 청크를 임베딩한 뒤, FAISS에 저장합니다.

## 이 단계가 하는 일

```text
documents.jsonl
-> context를 청크로 분할
-> 각 청크를 임베딩 벡터로 변환
-> FAISS 인덱스에 벡터 저장
-> 청크 원문과 출처는 metadata.jsonl에 저장
```

## 청크란?

청크는 긴 문서를 검색하기 좋게 자른 작은 조각입니다.

긴 context를 통째로 넣으면 질문과 관련 없는 내용까지 같이 섞일 수 있습니다.

너무 작게 자르면 정답 문장이 끊길 수 있습니다.

그래서 이 프로젝트에서는 청크 크기를 바꿔가며 실험합니다.

```text
300자 청크
700자 청크
```

## 임베딩이란?

임베딩은 문장을 숫자 배열로 바꾸는 작업입니다.

예를 들어 문장 하나가 다음처럼 변합니다.

```text
"이번 포럼의 주제는 청소년과 뉴미디어다."
-> [0.12, -0.03, 0.44, ...]
```

이 숫자 배열을 벡터라고 부릅니다.

벡터가 비슷하면 문장의 의미도 비슷하다고 판단합니다.

## 사용하는 모델

기본 모델은 다음입니다.

```text
intfloat/multilingual-e5-small
```

이 모델은 여러 언어를 지원하고, 질문과 문서 검색 실험에 쓰기 좋습니다.

E5 계열 모델은 입력 앞에 접두어를 붙여 쓰는 방식이 일반적입니다.

```text
문서 청크: passage: 문서 내용
질문: query: 질문 내용
```

## FAISS에는 무엇이 저장되나?

FAISS에는 벡터만 저장됩니다.

청크 원문, 문서 ID, 제목 같은 정보는 따로 `metadata.jsonl`에 저장합니다.

```text
index.faiss    = 벡터 검색용 인덱스
metadata.jsonl = 벡터가 어떤 청크에서 나왔는지 알려주는 정보
config.json    = 모델명, 청크 크기 같은 설정값
```

## 실행 방법

먼저 필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

300자 청크 인덱스:

```bash
python src/02_build_index.py --chunk-size 300
```

700자 청크 인덱스:

```bash
python src/02_build_index.py --chunk-size 700
```

생성 결과는 다음 폴더에 저장됩니다.

```text
vector_store/faiss/chunk_300/
vector_store/faiss/chunk_700/
```


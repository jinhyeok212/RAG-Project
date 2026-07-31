# 질문 입력
# → Retriever 검색
# → 평가
# → Generator 답변 생성
# → 화면 출력

from src.retriever import Retriever


def main() -> None:
    retriever = Retriever()

    question = input("질문을 입력하세요: ").strip()

    results = retriever.retrieve(
        question=question,
        top_k=3
    )

    print("\n검색 결과")

    for result in results:
        print("=" * 70)
        print(f"순위: {result['rank']}")
        print(f"유사도: {result['score']:.4f}")
        print(f"제목: {result['title']}")
        print(f"문서 ID: {result['doc_id']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"내용: {result['text']}")


if __name__ == "__main__":
    main()



# 검색 Chunk
# +
# 사용자 질문
# → Prompt
# → LLM
# → 최종 답변

import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def build_context(
    retrieved_chunks: list[dict]
) -> str:
    """
    검색된 Chunk를 LLM에 전달할 Context로 변환한다.
    """

    context_parts: list[str] = []

    for result in retrieved_chunks:
        context_parts.append(
            f"[문서 {result['rank']}]\n"
            f"제목: {result['title']}\n"
            f"내용: {result['text']}"
        )

    return "\n\n".join(context_parts)


def build_prompt(
    question: str,
    retrieved_chunks: list[dict]
) -> str:
    """
    질문과 검색 문서를 이용해 RAG Prompt를 만든다.
    """

    context = build_context(retrieved_chunks)

    return f"""
다음 제공 문서만 근거로 질문에 답하세요.

규칙:
1. 제공 문서에 없는 내용을 추측하지 마세요.
2. 답을 확인할 수 없다면
   "제공된 문서에서 답을 찾을 수 없습니다."라고 답하세요.
3. 답변은 한국어로 간결하게 작성하세요.
4. 답변 뒤에 근거가 된 문서 번호를 표시하세요.

제공 문서:
{context}

질문:
{question}

답변:
""".strip()


class Generator:
    """
    OpenAI API를 이용해 답변을 생성한다.
    """

    def __init__(
        self,
        model_name: str
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                ".env 파일에 OPENAI_API_KEY가 없습니다."
            )

        self.client = OpenAI(
            api_key=api_key
        )

        self.model_name = model_name

    def generate(
        self,
        question: str,
        retrieved_chunks: list[dict]
    ) -> dict:
        prompt = build_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks
        )

        response = self.client.responses.create(
            model=self.model_name,
            input=prompt
        )

        return {
            "answer": response.output_text,
            "prompt": prompt
        }
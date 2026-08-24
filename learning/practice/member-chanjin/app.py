from pathlib import Path

import pandas as pd
import streamlit as st

from src.rag_pipeline import append_logs, read_logs, read_markdown_documents, run_rag

BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge_base"
LOG_DIR = BASE_DIR / "data" / "logs"

st.set_page_config(page_title="RAG Core Chatbot", page_icon="🔎", layout="wide")

st.title("🔎 RAG Core Chatbot")
st.caption("팀원 과제용: RAG의 핵심 파이프라인과 파라미터를 직접 확인하는 미니 챗봇")

with st.sidebar:
    st.header("RAG 설정")
    st.markdown("문서를 나누고 검색하고 답변하는 과정을 파라미터별로 비교해보세요.")

    chunk_size = st.slider("chunk_size", min_value=250, max_value=1400, value=700, step=50)
    chunk_overlap = st.slider("chunk_overlap", min_value=0, max_value=400, value=120, step=20)
    if chunk_overlap >= chunk_size:
        st.warning("chunk_overlap은 chunk_size보다 작아야 합니다.")

    top_k = st.slider("top_k", min_value=1, max_value=10, value=4, step=1)
    similarity_threshold = st.slider("similarity_threshold", min_value=0.0, max_value=0.8, value=0.05, step=0.01)
    retrieval_mode_label = st.radio("retrieval_mode", ["Top-k", "MMR"], index=0)
    retrieval_mode = "mmr" if retrieval_mode_label == "MMR" else "top_k"
    mmr_lambda = st.slider("mmr_lambda", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

    retriever_backend_label = st.selectbox(
        "retriever_backend",
        ["TF-IDF 로컬 검색", "Sentence-Transformers 임베딩 검색(고급)"],
        index=0,
    )
    retriever_backend = "sentence-transformers" if "Sentence" in retriever_backend_label else "tfidf"
    embedding_model_name = st.text_input(
        "embedding_model_name",
        value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-Transformers 모드에서만 사용됩니다. 처음에는 TF-IDF 모드를 추천합니다.",
    )

    st.divider()
    st.header("생성 설정")
    model_mode_label = st.radio("answer_mode", ["API 키 없는 실습 모드", "OpenAI 생성 모드"], index=0)
    model_mode = "openai" if model_mode_label == "OpenAI 생성 모드" else "extractive"
    model_name = st.text_input("model_name", value="gpt-4o-mini")
    temperature = st.slider("temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    max_tokens = st.slider("max_tokens", min_value=200, max_value=2000, value=700, step=100)

    st.divider()
    with st.expander("파라미터 설명", expanded=False):
        st.markdown(
            """
            - **chunk_size**: 문서를 나누는 크기입니다. 작으면 정밀하지만 문맥 부족, 크면 문맥은 많지만 비용과 노이즈가 증가합니다.
            - **chunk_overlap**: chunk 간 겹치는 구간입니다. 경계에서 문맥이 끊기는 문제를 줄입니다.
            - **top_k**: 검색 결과를 몇 개 가져올지 정합니다. 크면 recall은 좋아질 수 있지만 비용과 노이즈가 증가합니다.
            - **similarity_threshold**: 최소 관련도 점수입니다. 높으면 관련 없는 문서를 줄일 수 있지만 필요한 문서도 제외될 수 있습니다.
            - **MMR**: 관련성뿐 아니라 다양성도 고려해 중복 chunk를 줄입니다.
            - **temperature**: 답변의 무작위성입니다. RAG는 근거 기반 답변이 중요하므로 보통 낮게 둡니다.
            - **max_tokens**: 답변 최대 길이입니다. 너무 크면 비용이 늘고, 너무 작으면 답변이 끊길 수 있습니다.
            - **논문 질문 팁**: Self-RAG, CRAG, HyDE, RAGAS처럼 논문명이나 기법명을 직접 넣으면 더 정확히 검색됩니다.
            """
        )


docs = read_markdown_documents(KNOWLEDGE_DIR)

col1, col2, col3 = st.columns(3)
col1.metric("지식 문서 수", len(docs))
col2.metric("검색 방식", retriever_backend_label)
col3.metric("답변 방식", model_mode_label)

with st.expander("현재 지식베이스 문서 보기", expanded=False):
    for doc in docs:
        st.markdown(f"- **{doc.title}** `{Path(doc.source_path).name}`")

st.subheader("질문하기")
example_questions = [
    "RAG가 뭐야?",
    "RAG 원논문의 핵심 아이디어가 뭐야?",
    "DPR과 TF-IDF 검색은 뭐가 달라?",
    "Self-RAG와 CRAG의 차이를 설명해줘.",
    "HyDE는 언제 쓰는 게 좋아?",
    "RAG-Fusion은 왜 multi-query를 사용해?",
    "RAPTOR와 GraphRAG는 어떤 상황에서 필요해?",
    "Lost in the Middle이 RAG에서 왜 중요한 문제야?",
    "RAGAS와 ARES는 각각 뭘 평가해?",
    "RAG에는 어떤 데이터가 필요해?",
    "chunk_size를 크게 하면 어떤 문제가 생겨?",
    "top_k는 왜 조정해야 해?",
    "similarity_threshold가 너무 높으면 어떻게 돼?",
    "RAG에서 temperature를 낮게 쓰는 이유는?",
    "검색 실패와 생성 실패는 어떻게 달라?",
    "LLMOps 대시보드에서 어떤 지표를 봐야 해?",
]
selected_example = st.selectbox("예시 질문", ["직접 입력"] + example_questions)
initial_question = "" if selected_example == "직접 입력" else selected_example
question = st.text_area("질문", value=initial_question, height=100, placeholder="예: top_k를 왜 조정해야 해?")

run_button = st.button("RAG 실행", type="primary", use_container_width=True)

if run_button:
    if not question.strip():
        st.error("질문을 입력해주세요.")
    elif chunk_overlap >= chunk_size:
        st.error("chunk_overlap은 chunk_size보다 작아야 합니다.")
    else:
        with st.spinner("문서 chunking → 검색 → 프롬프트 구성 → 답변 생성 → 로그 저장 중..."):
            try:
                result = run_rag(
                    question=question.strip(),
                    knowledge_dir=KNOWLEDGE_DIR,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    retriever_backend=retriever_backend,
                    embedding_model_name=embedding_model_name,
                    retrieval_mode=retrieval_mode,
                    mmr_lambda=mmr_lambda,
                    model_mode=model_mode,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                append_logs(result, LOG_DIR)
                st.success("RAG 실행 완료")

                st.subheader("답변")
                st.markdown(result.answer)

                st.subheader("검색된 문서")
                if not result.retrieved_chunks:
                    st.warning("검색된 문서가 없습니다. threshold를 낮추거나 top_k를 늘려보세요.")
                for item in result.retrieved_chunks:
                    with st.expander(f"#{item.rank} {item.title} | score={item.score:.4f}"):
                        st.write(item.text)
                        st.caption(f"chunk_id={item.chunk_id} / doc_id={item.doc_id} / chunk_index={item.chunk_index}")

                st.subheader("실행 로그 요약")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("전체 latency", f"{result.latency_ms} ms")
                m2.metric("검색 latency", f"{result.retrieval_latency_ms} ms")
                m3.metric("생성 latency", f"{result.generation_latency_ms} ms")
                m4.metric("예상 total tokens", result.total_tokens_est)

                with st.expander("LLM에 전달된 프롬프트 보기", expanded=False):
                    st.code(result.prompt, language="markdown")
            except Exception as e:
                st.exception(e)
                st.info("Sentence-Transformers 모드 오류라면 TF-IDF 로컬 검색으로 바꿔서 먼저 실행해보세요.")

st.divider()
st.subheader("로그 대시보드")
run_df, retrieval_df = read_logs(LOG_DIR)

if run_df.empty:
    st.info("아직 실행 로그가 없습니다. 질문을 실행하면 rag_run_logs.csv와 retrieval_logs.csv가 생성됩니다.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 실행 수", len(run_df))
    c2.metric("평균 latency", f"{run_df['latency_ms'].mean():.0f} ms")
    c3.metric("평균 예상 tokens", f"{run_df['total_tokens_est'].mean():.0f}")
    c4.metric("평균 top_k", f"{run_df['top_k'].mean():.1f}")

    tab1, tab2 = st.tabs(["RAG 실행 로그", "Retrieval 로그"])
    with tab1:
        st.dataframe(run_df.tail(20), use_container_width=True)
    with tab2:
        st.dataframe(retrieval_df.tail(50), use_container_width=True)

st.divider()
with st.expander("팀원 과제 안내", expanded=False):
    st.markdown(
        """
        1. 예시 질문 3개 이상을 실행합니다.  
        2. top_k를 3과 5로 바꿔 검색 문서가 어떻게 달라지는지 확인합니다.  
        3. chunk_size를 400과 900으로 바꿔 결과를 비교합니다.  
        4. similarity_threshold를 높였을 때 검색 결과가 사라지는지 확인합니다.  
        5. 실행 로그 CSV를 열어 어떤 값이 저장되는지 확인합니다.  
        6. 본인이 이해한 RAG 파이프라인을 5문장으로 정리합니다.  
        """
    )

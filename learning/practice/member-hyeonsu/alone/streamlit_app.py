"""Local Streamlit demo for the existing RAG service."""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.rag_service import RAGService
from src.retrieval_utils import load_environment

st.set_page_config(page_title="슈퍼 상담 RAG 챗봇", page_icon="💬", layout="wide")
load_environment()
# Safe deployment defaults: no .env or local Ollama is required for retrieval.
os.environ.setdefault("RAG_RETRIEVAL_STRATEGY", "intent_top3")
os.environ.setdefault("RAG_INTENT_TOP_N", "3")
os.environ.setdefault("RAG_INTENT_FALLBACK", "true")
os.environ.setdefault("RAG_DRY_RUN", "true")
os.environ.setdefault("RAG_COLLECTION", "super_questions")
os.environ.setdefault("CHROMA_PATH", "./chroma_db")
st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2rem;}
.hero {background: linear-gradient(135deg,#173b72,#2878b5); color:white; padding:2rem 2.2rem; border-radius:18px; margin-bottom:1.2rem; box-shadow:0 8px 24px #173b7226;}
.hero h1 {margin:0; font-size:2.1rem;} .hero p {margin:.5rem 0 0; opacity:.86;}
.answer-card {background:#f7f9fc; border:1px solid #e4eaf2; border-radius:14px; padding:1.1rem 1.3rem; line-height:1.75;}
.section-title {font-size:1.15rem; font-weight:700; margin:.8rem 0 .5rem; color:#173b72;}
[data-testid="stMetric"] {background:#f7f9fc; border:1px solid #e4eaf2; padding:.7rem; border-radius:12px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_service() -> RAGService:
    return RAGService()

@st.cache_resource
def get_intent_model(path: str):
    import joblib
    return joblib.load(path)

st.markdown('<div class="hero"><h1>💬 슈퍼 상담 RAG 챗봇</h1><p>검증된 상담 문서를 찾아 근거 기반으로 답변합니다. 기본 모드는 Retrieval-only입니다.</p></div>', unsafe_allow_html=True)
with st.sidebar:
    st.subheader("실행 옵션")
    retrieval_only = st.checkbox("Retrieval-only (LLM 호출 안 함)", value=True)
    top_k = st.number_input("검색 문서 수", min_value=1, max_value=20, value=int(os.getenv("RAG_TOP_K", "3")))
    st.info(f"검색 전략: {os.getenv('RAG_RETRIEVAL_STRATEGY', 'full')}")
    st.subheader("예시 질문")
    examples = [
        "배달도 되나요?",
        "고기 배달이 가능한가요?",
        "우유 가격이 얼마인가요?",
        "환불할 수 있나요?",
        "결제는 어떻게 하나요?",
        "카드 결제가 가능한가요?",
        "배송은 얼마나 걸리나요?",
        "상품을 교환할 수 있나요?",
        "재고가 있나요?",
        "매장은 몇 시에 문을 닫나요?",
    ]
    for example in examples:
        if st.button(example, use_container_width=True, key=f"example_{example}"):
            st.session_state["query_input"] = example
            st.rerun()

query = st.text_input("질문", placeholder="예: 우유는 얼마인가요?", label_visibility="collapsed", key="query_input")
if st.button("질문하기", type="primary"):
    if not query.strip():
        st.warning("질문을 입력해 주세요.")
        st.stop()
    try:
        service = get_service()
        model_path = str(ROOT / os.getenv("RAG_INTENT_MODEL_PATH", "models/super_intent_classifier.joblib"))
        model = get_intent_model(model_path)
        intent_started = time.perf_counter()
        probabilities = model.predict_proba([query])[0]
        classes = model.classes_
        order = probabilities.argsort()[::-1][:3]
        intent_ms = (time.perf_counter() - intent_started) * 1000

        # ask() reuses embedding, Chroma retrieval, prompt, LLM and JSONL logging.
        old_dry = service.dry_run
        if retrieval_only:
            service.dry_run = True
        result = service.ask(query)
        service.dry_run = old_dry

        st.markdown('<div class="section-title">최종 답변</div>', unsafe_allow_html=True)
        answer = "Retrieval-only 모드: LLM을 호출하지 않았습니다." if retrieval_only else result["final_answer"]
        st.markdown(f'<div class="answer-card">{answer.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">예측 인텐트 Top-3</div>', unsafe_allow_html=True)
        st.table([{"순위": i + 1, "인텐트": str(classes[idx]), "확률": f"{probabilities[idx]:.4f}"} for i, idx in enumerate(order)])

        with st.expander("검색 근거 및 세부 정보", expanded=True):
            docs = result.get("retrieved_documents", [])[:int(top_k)]
            if not docs:
                st.info("검색된 문서가 없습니다.")
            for doc in docs:
                st.markdown(f"**{doc.get('rank')}. {doc.get('retrieved_question','')}**")
                st.write(f"답변: {doc.get('retrieved_answer','')}")
                st.write(f"인텐트: {doc.get('intent','')} | Chroma distance: {doc.get('distance','N/A')}")
                st.divider()
        st.markdown('<div class="section-title">처리 시간</div>', unsafe_allow_html=True)
        t = result
        cols = st.columns(5)
        for col, label, value in zip(cols, ["인텐트 예측", "임베딩", "검색", "LLM 생성", "전체"], [intent_ms, t.get("query_embedding_time_ms", 0), t.get("retrieval_time_ms", 0), t.get("generation_time_ms", 0), t.get("total_time_ms", 0)]):
            col.metric(label, f"{value:.1f} ms")
        if result.get("error"):
            st.error(result["error"])
    except Exception as exc:
        st.error(f"RAG 실행 중 오류가 발생했습니다: {exc}")
        st.info("Ollama를 사용하는 경우 `ollama serve`가 실행 중인지 확인해 주세요.") #안녕

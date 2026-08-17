from __future__ import annotations

from pathlib import Path

import joblib

from src.common import load_config
from src.generation.generator import generate_answer
from src.retrieval.ann_index import DenseANNRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseExactRetriever
from src.retrieval.hybrid_retriever import HybridRetriever

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover
    raise SystemExit("streamlit is not installed. Run `pip install streamlit`.") from exc


@st.cache_resource
def load_system():
    cfg = load_config("configs/mvp.yaml")
    index_dir = Path(cfg["paths"]["index_dir"])
    bm25 = BM25Retriever.load(index_dir / "bm25.pkl")
    dense = DenseExactRetriever.load(index_dir / "dense_exact.pkl")
    ann = DenseANNRetriever.load(index_dir / "dense_ann.pkl") if (index_dir / "dense_ann.pkl").exists() else None
    docs = joblib.load(index_dir / "docs.joblib")
    return cfg, docs, HybridRetriever(bm25=bm25, dense=dense, ann=ann, docs=docs, cfg=cfg)


cfg, docs, retriever = load_system()
st.set_page_config(page_title="과거 상담 사례 RAG", layout="wide")
st.title("과거 상담 사례 검색 기반 RAG")

st.caption(
    "카페 카테고리 Train QA 20,000건을 검색해 유사한 과거 상담 답변을 근거로 응답하는 MVP입니다."
)

categories = sorted({doc.get("category") for doc in docs.values() if doc.get("category")})
intents = sorted({doc.get("intent") for doc in docs.values() if doc.get("intent")})

with st.sidebar:
    st.header("검색 파라미터")
    mode = st.selectbox("검색 방식", ["BM25", "Dense", "Hybrid", "Hybrid + Reranker"], index=3)
    top_k = st.slider("Top-k", 1, 10, 5, help="최종으로 화면에 보여줄 과거 상담 사례 수입니다.")
    candidate_k = st.slider(
        "Candidate-k",
        5,
        100,
        int(cfg["retrieval"]["reranker"].get("candidate_k", 50)),
        step=5,
        help="BM25/Dense에서 먼저 넓게 가져올 후보 수입니다. Reranker는 이 후보 안에서 다시 정렬합니다.",
    )
    filter_pool_k = st.slider(
        "Filter pool-k",
        100,
        5000,
        1000,
        step=100,
        help="카테고리/인텐트 필터를 적용하기 전에 넓게 가져올 후보 수입니다. 필터 결과가 없으면 이 값을 늘려보세요.",
    )
    use_ann = st.checkbox("Dense ANN 사용", value=False, help="Dense 검색을 Exact 대신 ANN fallback 인덱스로 수행합니다.")

    st.divider()
    st.subheader("Hybrid RRF")
    rrf_k = st.slider(
        "RRF k",
        10,
        120,
        int(cfg["retrieval"]["hybrid"].get("rrf_k", 60)),
        step=5,
        help="순위 결합에서 하위 결과의 영향력을 조절합니다. 값이 클수록 순위 차이가 완만해집니다.",
    )
    sparse_weight = st.slider(
        "BM25 가중치",
        0.0,
        3.0,
        float(cfg["retrieval"]["hybrid"].get("sparse_weight", 1.0)),
        step=0.1,
        help="키워드 기반 BM25 결과의 반영 비율입니다.",
    )
    dense_weight = st.slider(
        "Dense 가중치",
        0.0,
        3.0,
        float(cfg["retrieval"]["hybrid"].get("dense_weight", 1.0)),
        step=0.1,
        help="의미 기반 Dense 결과의 반영 비율입니다.",
    )

    st.divider()
    st.subheader("인덱스 재생성 파라미터")
    st.caption(
        "previous_context_turns, retrieval_text_version, max_train_docs는 corpus와 index를 다시 만들어야 반영됩니다."
    )
    st.code(
        "python -m src.data.build_corpus --config configs/mvp.yaml\n"
        "python -m src.retrieval.build_indexes --config configs/mvp.yaml",
        language="powershell",
    )

query = st.text_input("고객 질문", "아메리카노 주문 취소하고 싶어요")

col1, col2 = st.columns(2)
with col1:
    category_choice = st.selectbox("카테고리 필터", ["전체"] + categories)
    category = None if category_choice == "전체" else category_choice
with col2:
    intent_choice = st.selectbox("인텐트 필터", ["전체"] + intents)
    intent = None if intent_choice == "전체" else intent_choice

with st.expander("현재 파라미터 설명", expanded=False):
    st.markdown(
        f"""
- **Top-k = {top_k}**: 최종으로 보여줄 과거 상담 사례 수
- **Candidate-k = {candidate_k}**: 먼저 넓게 검색할 후보 수
- **Filter pool-k = {filter_pool_k}**: 필터 적용 전에 확보할 후보 수
- **RRF k = {rrf_k}**: BM25와 Dense 순위를 결합할 때 쓰는 완화 계수
- **BM25 가중치 = {sparse_weight:.1f}**: 상품명, 주문, 취소 같은 키워드 일치 영향
- **Dense 가중치 = {dense_weight:.1f}**: 의미적으로 비슷한 문장 검색 영향
- **Dense ANN 사용 = {use_ann}**: 빠른 근사 검색 fallback 사용 여부
"""
    )

if st.button("검색하고 답변 생성", type="primary"):
    kwargs = {
        "use_sparse": mode in {"BM25", "Hybrid", "Hybrid + Reranker"},
        "use_dense": mode in {"Dense", "Hybrid", "Hybrid + Reranker"},
        "use_reranker": mode == "Hybrid + Reranker",
        "use_ann": use_ann,
        "category": category,
        "intent": intent,
        "candidate_k": candidate_k,
        "filter_pool_k": filter_pool_k,
        "rrf_k": rrf_k,
        "sparse_weight": sparse_weight,
        "dense_weight": dense_weight,
    }
    results = retriever.search(query, top_k=top_k, **kwargs)
    answer = generate_answer(query, results, docs)

    st.subheader("생성 답변")
    st.write(answer["answer"])

    st.subheader("검색 근거")
    if not answer["sources"]:
        st.warning("검색 결과가 없습니다. 필터를 비우거나 Candidate-k/Top-k를 늘려보세요.")
    for item in answer["sources"]:
        marker = "답변 선택 / " if item.get("selected_for_answer") else ""
        title = f"{marker}{item['doc_id']} / {item.get('category')} / {item.get('intent')}"
        with st.expander(title):
            st.write("질문:", item["question"])
            st.write("답변:", item["answer"])
            st.json(item)

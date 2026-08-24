"""Core RAG pipeline for a small educational Streamlit chatbot.

This module intentionally keeps dependencies light. It supports:
- TF-IDF retrieval that runs locally without API keys.
- Optional Sentence-Transformers retrieval if the package/model is installed.
- Optional OpenAI generation if OPENAI_API_KEY is configured.

The goal is not to be the most powerful RAG system, but to expose the core
RAG steps clearly for team members: loading -> chunking -> indexing ->
retrieval -> prompt augmentation -> generation -> logging.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import time
import uuid
from dataclasses import dataclass, asdict
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Document:
    doc_id: str
    title: str
    source_path: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source_path: str
    chunk_index: int
    text: str


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    title: str
    chunk_index: int
    text: str
    score: float
    rank: int


@dataclass
class RagResult:
    run_id: str
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    prompt: str
    latency_ms: int
    retrieval_latency_ms: int
    generation_latency_ms: int
    retriever_backend: str
    retrieval_mode: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    similarity_threshold: float
    model_mode: str
    model_name: str
    temperature: float
    prompt_tokens_est: int
    completion_tokens_est: int
    total_tokens_est: int


def stable_id(text: str, prefix: str = "id") -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def read_markdown_documents(knowledge_dir: str | Path) -> List[Document]:
    knowledge_path = Path(knowledge_dir)
    docs: List[Document] = []
    for path in sorted(knowledge_path.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        title = extract_title(text) or path.stem
        doc_id = stable_id(str(path.name), "doc")
        docs.append(Document(doc_id=doc_id, title=title, source_path=str(path), text=text))
    return docs


def extract_title(markdown_text: str) -> Optional[str]:
    for line in markdown_text.splitlines():
        if line.startswith("# "):
            return line.replace("#", "", 1).strip()
    return None


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 120) -> List[str]:
    """Split text into overlapping character chunks.

    For first-time RAG practice, character-based chunking is easier to
    understand than tokenizer-based chunking. For production projects, use a
    tokenizer-aware splitter.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = clean_text(text)
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        # Try not to cut in the middle of a sentence/paragraph if possible.
        if end < len(text):
            last_break = max(chunk.rfind("\n\n"), chunk.rfind(". "), chunk.rfind("다. "))
            if last_break > chunk_size * 0.55:
                end = start + last_break + 1
                chunk = text[start:end]

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def build_chunks(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc in documents:
        for idx, chunk in enumerate(chunk_text(doc.text, chunk_size, chunk_overlap)):
            seed = f"{doc.doc_id}-{idx}-{chunk[:80]}"
            chunks.append(
                Chunk(
                    chunk_id=stable_id(seed, "chunk"),
                    doc_id=doc.doc_id,
                    title=doc.title,
                    source_path=doc.source_path,
                    chunk_index=idx,
                    text=chunk,
                )
            )
    return chunks


class TfidfRetriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            # Korean spacing and inflection make pure word TF-IDF brittle.
            # Character n-grams are a lightweight local fallback that can still
            # match terms like "top_k", "chunk_size", and Korean postpositions.
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
        )
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, top_k: int, similarity_threshold: float) -> List[Tuple[int, float]]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked_indices = np.argsort(scores)[::-1]
        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score < similarity_threshold:
                continue
            results.append((int(idx), score))
            if len(results) >= top_k:
                break
        return results

    def mmr_search(
        self,
        query: str,
        top_k: int,
        similarity_threshold: float,
        mmr_lambda: float = 0.7,
        candidate_pool_size: int = 20,
    ) -> List[Tuple[int, float]]:
        query_vec = self.vectorizer.transform([query])
        relevance = cosine_similarity(query_vec, self.matrix)[0]
        candidate_indices = [
            int(i)
            for i in np.argsort(relevance)[::-1]
            if float(relevance[i]) >= similarity_threshold
        ][: max(candidate_pool_size, top_k)]
        if not candidate_indices:
            return []

        selected: List[int] = []
        selected_scores: List[float] = []
        candidate_set = candidate_indices.copy()
        chunk_sim = cosine_similarity(self.matrix[candidate_indices], self.matrix[candidate_indices])
        idx_to_pos = {idx: pos for pos, idx in enumerate(candidate_indices)}

        while candidate_set and len(selected) < top_k:
            best_idx = None
            best_mmr = -float("inf")
            for idx in candidate_set:
                rel = float(relevance[idx])
                if not selected:
                    diversity_penalty = 0.0
                else:
                    idx_pos = idx_to_pos[idx]
                    selected_positions = [idx_to_pos[s] for s in selected]
                    diversity_penalty = float(np.max(chunk_sim[idx_pos, selected_positions]))
                mmr_score = mmr_lambda * rel - (1 - mmr_lambda) * diversity_penalty
                if mmr_score > best_mmr:
                    best_idx = idx
                    best_mmr = mmr_score
            if best_idx is None:
                break
            selected.append(best_idx)
            selected_scores.append(float(relevance[best_idx]))
            candidate_set.remove(best_idx)

        return list(zip(selected, selected_scores))


class SentenceTransformerRetriever:
    def __init__(self, chunks: List[Chunk], model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "sentence-transformers가 설치되어 있지 않습니다. "
                "pip install -r requirements-advanced.txt 후 다시 실행하세요."
            ) from exc

        self.chunks = chunks
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            [c.text for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def search(self, query: str, top_k: int, similarity_threshold: float) -> List[Tuple[int, float]]:
        q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = np.dot(self.embeddings, q)
        ranked_indices = np.argsort(scores)[::-1]
        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score < similarity_threshold:
                continue
            results.append((int(idx), score))
            if len(results) >= top_k:
                break
        return results

    def mmr_search(
        self,
        query: str,
        top_k: int,
        similarity_threshold: float,
        mmr_lambda: float = 0.7,
        candidate_pool_size: int = 20,
    ) -> List[Tuple[int, float]]:
        q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        relevance = np.dot(self.embeddings, q)
        candidate_indices = [
            int(i)
            for i in np.argsort(relevance)[::-1]
            if float(relevance[i]) >= similarity_threshold
        ][: max(candidate_pool_size, top_k)]
        if not candidate_indices:
            return []
        cand_emb = self.embeddings[candidate_indices]
        chunk_sim = np.dot(cand_emb, cand_emb.T)
        idx_to_pos = {idx: pos for pos, idx in enumerate(candidate_indices)}
        selected: List[int] = []
        selected_scores: List[float] = []
        candidate_set = candidate_indices.copy()
        while candidate_set and len(selected) < top_k:
            best_idx = None
            best_mmr = -float("inf")
            for idx in candidate_set:
                rel = float(relevance[idx])
                if not selected:
                    diversity_penalty = 0.0
                else:
                    idx_pos = idx_to_pos[idx]
                    selected_positions = [idx_to_pos[s] for s in selected]
                    diversity_penalty = float(np.max(chunk_sim[idx_pos, selected_positions]))
                mmr_score = mmr_lambda * rel - (1 - mmr_lambda) * diversity_penalty
                if mmr_score > best_mmr:
                    best_idx = idx
                    best_mmr = mmr_score
            if best_idx is None:
                break
            selected.append(best_idx)
            selected_scores.append(float(relevance[best_idx]))
            candidate_set.remove(best_idx)
        return list(zip(selected, selected_scores))


def build_retriever(chunks: List[Chunk], backend: str, embedding_model_name: str):
    if backend == "sentence-transformers":
        return SentenceTransformerRetriever(chunks, embedding_model_name)
    return TfidfRetriever(chunks)


@lru_cache(maxsize=12)
def get_cached_chunks_and_retriever(
    knowledge_dir_str: str,
    chunk_size: int,
    chunk_overlap: int,
    retriever_backend: str,
    embedding_model_name: str,
):
    """문서 chunk와 검색 인덱스를 캐싱합니다.

    Sentence-Transformers 모드는 모델 로딩과 문서 임베딩 생성이 느립니다.
    같은 설정으로 질문을 반복할 때는 이미 만든 retriever를 재사용해 속도를 줄입니다.
    지식 문서를 수정한 뒤에는 앱을 재시작해야 캐시가 갱신됩니다.
    """
    documents = read_markdown_documents(Path(knowledge_dir_str))
    chunks = build_chunks(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    retriever = build_retriever(chunks, backend=retriever_backend, embedding_model_name=embedding_model_name)
    return chunks, retriever


def estimate_tokens(text: str) -> int:
    """Very rough token estimate for dashboard practice.

    Korean tokenization varies by model. For a simple educational app, we use a
    coarse heuristic so team members can still see cost-related logging.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 2.2))


def build_prompt(question: str, retrieved_chunks: List[RetrievedChunk]) -> str:
    context_blocks = []
    for item in retrieved_chunks:
        context_blocks.append(
            f"[문서 {item.rank}] 제목: {item.title}\n"
            f"chunk_id: {item.chunk_id}\n"
            f"score: {item.score:.4f}\n"
            f"내용:\n{item.text}"
        )
    context = "\n\n---\n\n".join(context_blocks)
    return f"""당신은 RAG를 설명하는 교육용 챗봇입니다.
아래 검색 문서에 근거해서만 답변하세요.
문서에 없는 내용은 추측하지 말고, '제공된 문서만으로는 확실하지 않습니다'라고 말하세요.
답변 끝에는 참고한 문서 제목을 간단히 적으세요.

[검색 문서]
{context}

[사용자 질문]
{question}

[답변 지침]
- 한국어로 답변하세요.
- 처음 배우는 팀원도 이해할 수 있게 쉽게 설명하세요.
- 파라미터를 물어보면 '무엇인지', '왜 조정하는지', '너무 크거나 작을 때 문제'를 포함하세요.
""".strip()


def normalize_query_terms(question: str) -> str:
    """질문을 검색에 더 잘 걸리게 확장합니다.

    TF-IDF 로컬 검색은 LLM처럼 의미를 완전히 이해하지 못합니다. 그래서
    사용자가 '답변이 왜 똑같아?'처럼 물어도 관련 문서가 검색되도록
    RAG 주요 개념의 동의어/관련어를 질문 뒤에 붙입니다.
    """
    q = question.lower()
    expansions: List[str] = []
    term_map = {
        "rag": "retrieval augmented generation 검색 증강 생성 외부 문서 근거 답변",
        "데이터": "documents chunks eval_questions logs retrieval_logs rag_run_logs 지식베이스 질문 정답 근거문서",
        "chunk": "chunk_size chunk_overlap 청크 문서 분할 문맥 경계",
        "청크": "chunk_size chunk_overlap 문서 분할 문맥 경계",
        "top_k": "검색 결과 개수 recall noise 비용 정답 문서",
        "threshold": "similarity_threshold 유사도 기준 검색 제외 관련 없는 문서",
        "유사도": "similarity_threshold score cosine 관련도 검색 기준",
        "mmr": "diversity redundancy 중복 제거 다양성 mmr_lambda",
        "lambda": "mmr_lambda 관련성 다양성 균형",
        "embedding": "임베딩 벡터 의미 유사도 sentence-transformers",
        "임베딩": "embedding 벡터 의미 유사도 sentence-transformers",
        "retriever": "검색기 vector search bm25 hybrid reranker",
        "검색": "retriever retrieval vector search bm25 hybrid reranker",
        "reranker": "재정렬 cross encoder 상위 문서 정확도",
        "temperature": "생성 무작위성 일관성 근거 충실도 낮게",
        "max_tokens": "답변 길이 비용 토큰 제한",
        "hallucination": "환각 근거 없는 답변 faithfulness groundedness",
        "환각": "hallucination 근거 없는 답변 faithfulness groundedness",
        "평가": "faithfulness answer relevancy context precision context recall hit rate mrr",
        "llmops": "품질 비용 latency token monitoring dashboard 운영 로그",
        "로그": "rag_run_logs retrieval_logs latency token cost evaluation score",
        "검색 실패": "retrieval failure 정답 문서 미검색 top_k threshold embedding chunk",
        "생성 실패": "generation failure 문서는 찾았지만 답변 오류 hallucination prompt",
        "논문": "paper survey REALM DPR RAG FiD KILT GAR Atlas REPLUG HyDE FLARE Self-RAG CRAG RAG-Fusion RAPTOR GraphRAG RAGAS ARES Lost in the Middle",
        "paper": "논문 survey REALM DPR RAG FiD KILT GAR Atlas REPLUG HyDE FLARE Self-RAG CRAG RAG-Fusion RAPTOR GraphRAG RAGAS ARES",
        "realm": "REALM retrieval augmented language model pretraining retriever masked language model",
        "dpr": "Dense Passage Retrieval dense retriever dual encoder sparse retrieval BM25 TF-IDF",
        "fid": "Fusion-in-Decoder passage retrieval open domain question answering top_k multiple passages",
        "kilt": "Knowledge Intensive Language Tasks provenance grounding Wikipedia benchmark",
        "hyde": "Hypothetical Document Embeddings query transformation zero-shot dense retrieval",
        "flare": "Forward-Looking Active Retrieval active retrieval low confidence generation",
        "self-rag": "Self-RAG self reflection retrieve generate critique reflection tokens",
        "crag": "Corrective Retrieval Augmented Generation retrieval evaluator correction fallback web search",
        "rag-fusion": "RAG-Fusion multi query Reciprocal Rank Fusion RRF query rewriting",
        "raptor": "RAPTOR recursive abstractive processing tree organized retrieval hierarchical summaries",
        "graphrag": "GraphRAG graph RAG entity community summary local global query focused summarization",
        "ragas": "RAGAS context precision context recall faithfulness answer relevancy automated evaluation",
        "ares": "ARES automated evaluation context relevance answer faithfulness answer relevance LM judge",
        "lost in the middle": "long context position bias relevant information middle context RAG",
    }
    for key, value in term_map.items():
        if key in q:
            expansions.append(value)
    return question + (" " + " ".join(expansions) if expansions else "")


def detect_question_intent(question: str) -> str:
    q = question.lower()
    paper_terms = [
        "논문", "paper", "survey", "realm", "dpr", "fid", "kilt", "hyde", "flare",
        "self-rag", "self rag", "crag", "rag-fusion", "rag fusion", "raptor",
        "graphrag", "graph rag", "ragas", "ares", "lost in the middle", "lewis"
    ]
    if any(x in q for x in paper_terms):
        return "rag_paper"
    if any(x in q for x in ["chunk", "청크", "chunk_size", "chunk overlap", "chunk_overlap"]):
        return "chunking"
    if any(x in q for x in ["top_k", "top k", "top-k"]):
        return "top_k"
    if any(x in q for x in ["threshold", "유사도", "similarity"]):
        return "threshold"
    if "mmr" in q or "lambda" in q or "다양" in q or "중복" in q:
        return "mmr"
    if any(x in q for x in ["temperature", "max_tokens", "토큰", "생성"]):
        return "generation_params"
    if any(x in q for x in ["embedding", "임베딩", "벡터"]):
        return "embedding"
    if any(x in q for x in ["평가", "지표", "llmops", "대시보드", "모니터링", "품질"]):
        return "evaluation_llmops"
    if any(x in q for x in ["데이터", "dataset", "데이터셋", "문서", "로그"]):
        return "data"
    if "검색 실패" in q or "생성 실패" in q or "실패" in q:
        return "failure"
    if "rag" in q or "알려" in q or "뭐" in q:
        return "rag_overview"
    return "general"


def intent_intro(intent: str, question: str) -> str:
    intros = {
        "chunking": (
            "질문은 **Chunking(문서 분할)**에 대한 내용입니다. Chunking은 긴 문서를 검색 가능한 작은 단위로 나누는 단계이고, "
            "`chunk_size`와 `chunk_overlap` 설정에 따라 검색 품질과 비용이 달라집니다."
        ),
        "top_k": (
            "질문은 **top_k**에 대한 내용입니다. top_k는 검색된 후보 문서 중 LLM에게 넘길 상위 문서 개수입니다. "
            "값을 키우면 정답 문서를 포함할 가능성은 올라가지만, 노이즈와 토큰 비용도 함께 늘 수 있습니다."
        ),
        "threshold": (
            "질문은 **similarity_threshold**에 대한 내용입니다. 이 값은 검색 결과로 인정할 최소 유사도 기준입니다. "
            "너무 높으면 필요한 문서도 걸러지고, 너무 낮으면 관련 없는 문서가 답변에 섞입니다."
        ),
        "mmr": (
            "질문은 **MMR(Maximal Marginal Relevance)**에 대한 내용입니다. MMR은 관련도만 보는 것이 아니라, "
            "비슷한 문서가 반복 검색되는 문제를 줄이기 위해 다양성도 함께 고려합니다."
        ),
        "generation_params": (
            "질문은 **생성 파라미터**에 대한 내용입니다. RAG에서는 창의적인 답변보다 근거에 충실한 답변이 중요하므로, "
            "temperature와 max_tokens를 목적에 맞게 조정해야 합니다."
        ),
        "embedding": (
            "질문은 **Embedding과 검색**에 대한 내용입니다. 임베딩은 문장이나 문서를 벡터로 바꿔 질문과 문서의 의미적 유사도를 계산하게 해줍니다."
        ),
        "evaluation_llmops": (
            "질문은 **RAG 평가와 LLMOps**에 대한 내용입니다. RAG는 답변이 나오는지만 보는 것이 아니라, "
            "검색 품질·근거 충실도·응답 속도·비용을 함께 모니터링해야 합니다."
        ),
        "data": (
            "질문은 **RAG에 필요한 데이터**에 대한 내용입니다. 기본적으로 지식 문서, 문서 청크, 평가 질문, 정답 근거, 실행 로그가 필요합니다."
        ),
        "failure": (
            "질문은 **RAG 실패 원인**에 대한 내용입니다. RAG 실패는 크게 검색 실패와 생성 실패로 나눠서 봐야 합니다."
        ),
        "rag_overview": (
            "질문은 **RAG의 기본 개념**에 대한 내용입니다. RAG는 모델이 기억한 지식만 쓰지 않고, 외부 문서를 먼저 검색한 뒤 그 근거로 답변하는 방식입니다."
        ),
        "rag_paper": (
            "질문은 **RAG 관련 논문/기법**에 대한 내용입니다. 아래 답변은 검색된 논문 요약 노트를 바탕으로, "
            "논문이 해결하려는 문제와 우리 프로젝트에 적용할 수 있는 포인트를 중심으로 정리합니다."
        ),
        "general": "검색된 문서를 기준으로 질문에 가장 관련 있는 내용을 정리했습니다.",
    }
    return intros.get(intent, intros["general"])


def intent_parameter_guide(intent: str) -> List[str]:
    guides = {
        "chunking": [
            "`chunk_size`를 작게 하면 검색은 정밀해지지만 문맥이 부족해질 수 있습니다.",
            "`chunk_size`를 크게 하면 문맥은 충분해지지만 관련 없는 내용이 섞이고 토큰 비용이 늘 수 있습니다.",
            "`chunk_overlap`은 문단 경계에서 중요한 설명이 잘리는 문제를 줄이기 위해 사용합니다.",
        ],
        "top_k": [
            "`top_k`가 너무 작으면 정답 근거 문서를 놓칠 수 있습니다.",
            "`top_k`가 너무 크면 관련 없는 문서까지 들어가 답변이 흐려지고 비용이 증가합니다.",
            "실험에서는 보통 top_k=3, 5, 10을 비교해 hit rate와 비용의 균형점을 찾습니다.",
        ],
        "threshold": [
            "threshold를 높이면 검색 결과의 정밀도는 올라갈 수 있지만 recall은 떨어질 수 있습니다.",
            "threshold를 낮추면 더 많은 문서를 가져오지만 노이즈가 늘어납니다.",
            "검색 결과가 자주 비어 있다면 threshold를 낮추고, 엉뚱한 문서가 많다면 threshold를 높여봅니다.",
        ],
        "mmr": [
            "`mmr_lambda`가 1에 가까우면 관련도를 더 중시합니다.",
            "`mmr_lambda`가 0에 가까우면 다양성을 더 중시합니다.",
            "비슷한 chunk가 반복 검색될 때 MMR을 사용하면 근거 문서 구성이 더 넓어질 수 있습니다.",
        ],
        "generation_params": [
            "`temperature`를 낮추면 답변이 안정적이고 일관되지만 덜 창의적입니다.",
            "RAG에서는 근거 기반 답변이 중요하므로 보통 temperature를 낮게 둡니다.",
            "`max_tokens`가 너무 작으면 답변이 끊기고, 너무 크면 비용이 늘 수 있습니다.",
        ],
        "evaluation_llmops": [
            "검색 단계는 hit rate, MRR, context precision/recall로 평가합니다.",
            "생성 단계는 faithfulness, answer relevancy, hallucination rate로 평가합니다.",
            "운영 단계는 latency, token 수, cost, error rate, bad feedback rate를 모니터링합니다.",
        ],
        "data": [
            "문서 데이터는 RAG가 검색할 지식베이스입니다.",
            "평가 질문과 정답 근거 문서는 RAG가 제대로 검색·답변했는지 평가하는 데 필요합니다.",
            "실행 로그는 LLMOps 대시보드에서 품질·비용·속도를 분석하는 핵심 데이터입니다.",
        ],
        "failure": [
            "검색 실패는 정답 근거 문서를 못 찾은 경우입니다.",
            "생성 실패는 문서는 찾았지만 답변이 질문과 맞지 않거나 근거를 왜곡한 경우입니다.",
            "두 실패를 분리해야 chunking, retrieval, prompt 중 어디를 고쳐야 하는지 알 수 있습니다.",
        ],
        "rag_paper": [
            "논문을 볼 때는 '어떤 RAG 실패를 해결하려는가'를 먼저 봅니다.",
            "구현까지 하지 않더라도 논문 아이디어를 top_k, chunk_size, reranker, query rewriting, evaluation metric 같은 실험 변수로 바꿀 수 있습니다.",
            "우리 프로젝트에서는 논문별 기법보다 품질·비용·지연시간 지표로 연결하는 해석이 중요합니다.",
        ],
    }
    return guides.get(intent, [])


def score_sentence_for_question(sentence: str, question_keywords: set[str], intent: str) -> int:
    tokens = set(re.findall(r"[가-힣A-Za-z0-9_]{2,}", sentence.lower()))
    score = len(question_keywords & tokens) * 3
    intent_terms = {
        "chunking": ["chunk", "청크", "overlap", "문서", "분할", "문맥"],
        "top_k": ["top_k", "검색", "문서", "개수", "비용", "노이즈"],
        "threshold": ["threshold", "유사도", "관련", "제외", "검색"],
        "mmr": ["mmr", "다양", "중복", "관련", "lambda"],
        "generation_params": ["temperature", "max_tokens", "생성", "토큰", "답변"],
        "embedding": ["embedding", "임베딩", "벡터", "유사도", "의미"],
        "evaluation_llmops": ["평가", "지표", "faithfulness", "latency", "비용", "모니터링"],
        "data": ["데이터", "문서", "질문", "로그", "정답", "근거"],
        "failure": ["실패", "검색", "생성", "정답", "근거"],
        "rag_overview": ["rag", "검색", "생성", "외부", "문서", "근거"],
        "rag_paper": ["논문", "paper", "realm", "dpr", "fid", "hyde", "flare", "self-rag", "crag", "raptor", "graphrag", "ragas", "ares", "평가", "검색", "생성"],
    }
    for term in intent_terms.get(intent, []):
        if term.lower() in sentence.lower():
            score += 2
    return score


def generate_extractive_answer(question: str, retrieved_chunks: List[RetrievedChunk]) -> str:
    """API key 없이 동작하는 실습용 답변 생성기.

    이전 버전은 단순히 상위 문서의 문장을 가져와서 질문이 달라도 답변이 비슷하게 보일 수 있었습니다.
    개선 버전은 질문 의도를 먼저 분류하고, 의도별 설명 + 파라미터 조정 이유 + 검색 근거를 조합합니다.
    """
    if not retrieved_chunks:
        return (
            "관련 문서를 찾지 못했습니다. similarity_threshold를 낮추거나 top_k를 늘려보세요. "
            "또는 지식베이스에 질문과 관련된 문서가 있는지 확인해야 합니다."
        )

    intent = detect_question_intent(question)
    question_keywords = set(re.findall(r"[가-힣A-Za-z0-9_]{2,}", normalize_query_terms(question).lower()))

    candidate_sentences: List[Tuple[int, int, str]] = []
    for chunk in retrieved_chunks:
        split = re.split(r"(?<=[.!?다요])\s+|\n+", chunk.text)
        for sent in split:
            sent = sent.strip(" -\n\t")
            if len(sent) < 18:
                continue
            if sent.startswith("#") or "예시 질문" in sent or "입력해본다" in sent:
                continue
            score = score_sentence_for_question(sent, question_keywords, intent)
            if score <= 0:
                continue
            # rank가 높을수록 약간 가산점
            score += max(0, 4 - chunk.rank)
            candidate_sentences.append((score, len(sent), sent))

    candidate_sentences.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    evidence: List[str] = []
    for _, _, sent in candidate_sentences:
        if sent not in evidence:
            evidence.append(sent)
        if len(evidence) >= 3:
            break

    if not evidence:
        evidence = [retrieved_chunks[0].text[:450].replace("\n", " ")]

    sources = ", ".join(dict.fromkeys([c.title for c in retrieved_chunks[:3]]))
    guide = intent_parameter_guide(intent)

    parts = [
        "API 키 없는 실습 모드입니다. 아래 답변은 LLM 생성이 아니라, 질문 의도 분류와 검색 문서 문장 추출로 구성한 답변입니다.",
        "",
        f"### 핵심 답변\n{intent_intro(intent, question)}",
    ]

    if guide:
        parts.append("\n### 파라미터 조정 관점")
        parts.extend([f"- {g}" for g in guide])

    parts.append("\n### 검색 문서에서 확인한 근거")
    parts.extend([f"- {s}" for s in evidence])
    parts.append(f"\n참고 문서: {sources}")
    return "\n".join(parts)

def generate_openai_answer(prompt: str, model_name: str, temperature: float, max_tokens: int) -> str:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("openai 패키지가 설치되어 있지 않습니다. pip install openai를 실행하세요.") from exc

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "당신은 근거 기반으로 답하는 한국어 RAG 교육용 챗봇입니다."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def run_rag(
    question: str,
    knowledge_dir: str | Path,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    top_k: int = 4,
    similarity_threshold: float = 0.05,
    retriever_backend: str = "tfidf",
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    retrieval_mode: str = "top_k",
    mmr_lambda: float = 0.7,
    model_mode: str = "extractive",
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.1,
    max_tokens: int = 700,
) -> RagResult:
    start = time.perf_counter()

    chunks, retriever = get_cached_chunks_and_retriever(
        str(Path(knowledge_dir).resolve()),
        chunk_size,
        chunk_overlap,
        retriever_backend,
        embedding_model_name,
    )

    # 로컬 TF-IDF 검색은 LLM처럼 의미를 추론하지 못하기 때문에,
    # 질문에 RAG 관련 동의어/핵심어를 확장해서 검색 품질을 높입니다.
    search_query = normalize_query_terms(question)

    retrieval_start = time.perf_counter()
    if retrieval_mode == "mmr":
        raw_results = retriever.mmr_search(
            search_query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            mmr_lambda=mmr_lambda,
        )
    else:
        raw_results = retriever.search(search_query, top_k=top_k, similarity_threshold=similarity_threshold)
    retrieval_latency_ms = int((time.perf_counter() - retrieval_start) * 1000)

    retrieved_chunks = [
        RetrievedChunk(
            chunk_id=chunks[idx].chunk_id,
            doc_id=chunks[idx].doc_id,
            title=chunks[idx].title,
            chunk_index=chunks[idx].chunk_index,
            text=chunks[idx].text,
            score=score,
            rank=rank,
        )
        for rank, (idx, score) in enumerate(raw_results, start=1)
    ]
    prompt = build_prompt(question, retrieved_chunks)

    generation_start = time.perf_counter()
    if model_mode == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            answer = (
                "OPENAI_API_KEY가 설정되지 않아 OpenAI 생성 모드를 사용할 수 없습니다. "
                "현재는 실습용 추출 답변으로 대체합니다.\n\n"
                + generate_extractive_answer(question, retrieved_chunks)
            )
        else:
            answer = generate_openai_answer(prompt, model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    else:
        answer = generate_extractive_answer(question, retrieved_chunks)
    generation_latency_ms = int((time.perf_counter() - generation_start) * 1000)

    latency_ms = int((time.perf_counter() - start) * 1000)
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = estimate_tokens(answer)

    return RagResult(
        run_id=str(uuid.uuid4()),
        question=question,
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        prompt=prompt,
        latency_ms=latency_ms,
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
        retriever_backend=retriever_backend,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        similarity_threshold=similarity_threshold,
        model_mode=model_mode,
        model_name=model_name,
        temperature=temperature,
        prompt_tokens_est=prompt_tokens,
        completion_tokens_est=completion_tokens,
        total_tokens_est=prompt_tokens + completion_tokens,
    )


def append_logs(result: RagResult, log_dir: str | Path) -> None:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    run_log_file = log_path / "rag_run_logs.csv"
    retrieval_log_file = log_path / "retrieval_logs.csv"

    run_fields = [
        "run_time",
        "run_id",
        "question",
        "answer",
        "latency_ms",
        "retrieval_latency_ms",
        "generation_latency_ms",
        "retriever_backend",
        "retrieval_mode",
        "top_k",
        "chunk_size",
        "chunk_overlap",
        "similarity_threshold",
        "model_mode",
        "model_name",
        "temperature",
        "prompt_tokens_est",
        "completion_tokens_est",
        "total_tokens_est",
    ]
    run_row = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "run_id": result.run_id,
        "question": result.question,
        "answer": result.answer,
        "latency_ms": result.latency_ms,
        "retrieval_latency_ms": result.retrieval_latency_ms,
        "generation_latency_ms": result.generation_latency_ms,
        "retriever_backend": result.retriever_backend,
        "retrieval_mode": result.retrieval_mode,
        "top_k": result.top_k,
        "chunk_size": result.chunk_size,
        "chunk_overlap": result.chunk_overlap,
        "similarity_threshold": result.similarity_threshold,
        "model_mode": result.model_mode,
        "model_name": result.model_name,
        "temperature": result.temperature,
        "prompt_tokens_est": result.prompt_tokens_est,
        "completion_tokens_est": result.completion_tokens_est,
        "total_tokens_est": result.total_tokens_est,
    }
    write_header = not run_log_file.exists()
    with run_log_file.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=run_fields)
        if write_header:
            writer.writeheader()
        writer.writerow(run_row)

    retrieval_fields = [
        "run_time",
        "run_id",
        "question",
        "rank",
        "chunk_id",
        "doc_id",
        "title",
        "chunk_index",
        "score",
        "retriever_backend",
        "retrieval_mode",
    ]
    write_header = not retrieval_log_file.exists()
    with retrieval_log_file.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=retrieval_fields)
        if write_header:
            writer.writeheader()
        for item in result.retrieved_chunks:
            writer.writerow(
                {
                    "run_time": datetime.now().isoformat(timespec="seconds"),
                    "run_id": result.run_id,
                    "question": result.question,
                    "rank": item.rank,
                    "chunk_id": item.chunk_id,
                    "doc_id": item.doc_id,
                    "title": item.title,
                    "chunk_index": item.chunk_index,
                    "score": item.score,
                    "retriever_backend": result.retriever_backend,
                    "retrieval_mode": result.retrieval_mode,
                }
            )


def read_logs(log_dir: str | Path) -> Tuple[Any, Any]:
    import pandas as pd

    log_path = Path(log_dir)
    run_log_file = log_path / "rag_run_logs.csv"
    retrieval_log_file = log_path / "retrieval_logs.csv"
    run_df = pd.read_csv(run_log_file) if run_log_file.exists() else pd.DataFrame()
    retrieval_df = pd.read_csv(retrieval_log_file) if retrieval_log_file.exists() else pd.DataFrame()
    return run_df, retrieval_df

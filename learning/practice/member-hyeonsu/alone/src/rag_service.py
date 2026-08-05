"""기존 검색 baseline과 LLM 생성을 연결하는 최소 RAG 서비스."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
import joblib

from src.chroma_store import ChromaQuestionStore
from src.embedding_model import KoreanEmbeddingModel
from src.llm_client import LLMClient, LLMClientError, load_llm_config
from src.rag_prompt import build_rag_prompt
from src.retrieval_utils import env_int, load_environment, project_path

DRY_RUN_ANSWER = "DRY RUN 모드이므로 LLM을 호출하지 않았습니다."
NO_RESULTS_ANSWER = "관련 상담 자료를 충분히 찾지 못했습니다.\n질문을 조금 더 구체적으로 입력해주세요."
EMPTY_ANSWER = "제공된 상담 자료만으로는 확인하기 어렵습니다."
CONFLICT_ANSWER = (
    "검색된 상담 자료마다 답변이 달라 정확한 안내가 어렵습니다.\n"
    "확인하려는 상품명이나 매장 정보를 알려주세요."
)
AMBIGUOUS_ANSWER = (
    "질문의 대상이나 상황을 확인하기 어렵습니다.\n"
    "확인하려는 상품명이나 매장 정보를 조금 더 구체적으로 알려주세요."
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    if raw in {"true", "1", "yes", "on"}:
        return True
    if raw in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"{name}은 true 또는 false여야 합니다.")


def _optional_distance() -> float | None:
    raw = os.getenv("RAG_MAX_DISTANCE", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("RAG_MAX_DISTANCE는 숫자이거나 빈 값이어야 합니다.") from exc
    if value < 0:
        raise ValueError("RAG_MAX_DISTANCE는 0 이상이어야 합니다.")
    return value


def _llm_identity() -> tuple[str, str]:
    """API 키를 읽거나 검증하지 않고 로그용 provider와 모델명만 확인합니다."""
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider == "ollama":
        return provider, os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()
    if provider == "openai":
        return provider, os.getenv("OPENAI_MODEL", "").strip()
    return provider, ""


def _is_context_ambiguous(query: str) -> bool:
    """대상을 가리키는 말만 있고 상품 식별 정보가 없는 대표 패턴을 찾습니다."""
    compact = re.sub(r"\s+", "", query)
    demonstrative = re.search(r"(?:이거|그거|저거|이것|그것|저것)", compact)
    broad_request = re.search(r"(?:얼마|있어|있나요|언제|뭐|어때|되나요|가능)", compact)
    return bool(demonstrative and broad_request and len(compact) <= 20)


def _fact_values(answer: str, fact_type: str) -> set[str]:
    """답변에 명시된 가격·수량·날짜/시간 후보를 비교 가능한 문자열로 추출합니다."""
    if fact_type == "price":
        values: set[str] = set()
        for match in re.finditer(r"(\d[\d,]*)\s*(만원|천원|원)", answer):
            number = int(match.group(1).replace(",", ""))
            multiplier = {"원": 1, "천원": 1_000, "만원": 10_000}[match.group(2)]
            values.add(f"{number * multiplier}원")
        if "무료" in answer:
            values.add("무료")
        return values
    elif fact_type == "quantity":
        pattern = r"\d+\s*(?:개|병|봉|박스|팩|세트|kg|g|리터|L|인분)"
    else:
        pattern = r"\d+\s*(?:시|분|일|월|년)|오늘|내일|모레|오전|오후"
    return {re.sub(r"\s+", "", value).lower() for value in re.findall(pattern, answer, re.I)}


def _has_conflicting_facts(query: str, documents: list[dict[str, Any]]) -> bool:
    """사용자가 물은 사실 유형에서 서로 다른 명시적 값이 검색됐는지 보수적으로 확인합니다."""
    fact_types = []
    if re.search(r"가격|얼마|비용|요금", query):
        fact_types.append("price")
    if re.search(r"수량|몇\s*(?:개|병|봉|박스)|개수", query):
        fact_types.append("quantity")
    if re.search(r"언제|날짜|시간|몇\s*시|도착|입고", query):
        fact_types.append("datetime")
    for fact_type in fact_types:
        values: set[str] = set()
        for document in documents:
            values.update(_fact_values(document["retrieved_answer"], fact_type))
        if len(values) > 1:
            return True
    return False


class RAGService:
    """모델과 DB는 한 번만 로드하고 질문마다 검색과 생성을 수행합니다."""

    def __init__(self) -> None:
        load_environment()
        self.top_k = env_int("RAG_TOP_K", 3)
        self.collection_name = os.getenv("RAG_COLLECTION", "super_questions").strip()
        if not self.collection_name:
            raise ValueError("RAG_COLLECTION이 비어 있습니다.")
        self.max_distance = _optional_distance()
        self.debug_prompt = _env_bool("RAG_DEBUG_PROMPT")
        self.dry_run = _env_bool("RAG_DRY_RUN")
        self.retrieval_strategy = os.getenv("RAG_RETRIEVAL_STRATEGY", "full").strip().lower()
        self.intent_top_n = env_int("RAG_INTENT_TOP_N", 3)
        self.intent_fallback = _env_bool("RAG_INTENT_FALLBACK", True)
        self.log_path = project_path("results/rag_generation/rag_chat_log.jsonl")

        # 기존 store는 없는 컬렉션을 만들기 때문에 먼저 읽기 전용 목록으로 존재를 검사합니다.
        chroma_path = project_path(os.getenv("CHROMA_PATH", "./chroma_db"))
        client = chromadb.PersistentClient(path=str(chroma_path))
        names = {item.name for item in client.list_collections()}
        if self.collection_name not in names:
            raise RuntimeError(
                f"Chroma 컬렉션 '{self.collection_name}'이 없습니다. 먼저 기존 인덱스를 확인해주세요."
            )
        self.store = ChromaQuestionStore(
            path=str(chroma_path), collection_name=self.collection_name
        )
        self.intent_model = None
        if self.retrieval_strategy in {"intent_top1", "intent_top3"}:
            model_path = project_path(os.getenv("RAG_INTENT_MODEL_PATH", "models/super_intent_classifier.joblib"))
            if not model_path.exists():
                raise RuntimeError(f"의도 분류 모델을 찾을 수 없습니다: {model_path}")
            self.intent_model = joblib.load(model_path)
        try:
            self.embedding_model = KoreanEmbeddingModel()
        except Exception as exc:
            raise RuntimeError(f"임베딩 모델을 불러오지 못했습니다: {exc}") from exc

    def _normalize_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        documents = []
        for result in results:
            metadata = result.get("metadata") or {}
            documents.append(
                {
                    "rank": result.get("rank"),
                    "document_id": result.get("document_id", ""),
                    "retrieved_question": result.get("question", ""),
                    "retrieved_answer": str(metadata.get("answer", "") or ""),
                    "intent": str(metadata.get("intent", "") or ""),
                    "category": str(metadata.get("category", "") or ""),
                    "distance": result.get("distance"),
                }
            )
        return documents

    def ask(self, user_query: str) -> dict[str, Any]:
        started = time.perf_counter()
        provider, model = _llm_identity()
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_query": user_query,
            "llm_provider": provider,
            "llm_model": model,
            "retrieved_documents": [],
            "final_answer": "",
            "query_embedding_time_ms": 0.0,
            "retrieval_time_ms": 0.0,
            "generation_time_ms": 0.0,
            "total_time_ms": 0.0,
            "error": None,
            "prompt": "",
        }
        try:
            embed_started = time.perf_counter()
            embedding = self.embedding_model.encode([user_query])[0]
            record["query_embedding_time_ms"] = (time.perf_counter() - embed_started) * 1000

            retrieval_started = time.perf_counter()
            query_results = self.store.query(embedding, self.top_k)
            if self.intent_model is not None:
                intents = list(self.intent_model.classes_)
                predicted = list(self.intent_model.predict_proba([user_query])[0])
                ranked = [intents[i] for i in sorted(range(len(intents)), key=lambda i: predicted[i], reverse=True)[: (1 if self.retrieval_strategy == "intent_top1" else self.intent_top_n)]]
                filtered = []
                for intent in ranked:
                    filtered.extend(self.store.query(embedding, self.top_k, where={"intent": intent}))
                query_results = sorted({d["document_id"]: d for d in filtered}.values(), key=lambda d: d["distance"])[: self.top_k]
                if not query_results and self.intent_fallback:
                    query_results = self.store.query(embedding, self.top_k)
            documents = self._normalize_results(query_results)
            record["retrieval_time_ms"] = (time.perf_counter() - retrieval_started) * 1000
            if self.max_distance is not None:
                documents = [
                    d for d in documents
                    if d.get("distance") is not None and d["distance"] <= self.max_distance
                ]
            record["retrieved_documents"] = documents

            if not documents:
                record["final_answer"] = NO_RESULTS_ANSWER
            elif not any(d["retrieved_answer"].strip() for d in documents):
                record["final_answer"] = EMPTY_ANSWER
            else:
                # 답변이 빈 일부 문서는 LLM 근거에서 제외합니다.
                documents = [d for d in documents if d["retrieved_answer"].strip()]
                record["retrieved_documents"] = documents
                prompt = build_rag_prompt(user_query, documents)
                record["prompt"] = prompt
                if self.debug_prompt:
                    print("\n[LLM 전달 프롬프트]\n" + prompt)
                if self.dry_run:
                    record["final_answer"] = DRY_RUN_ANSWER
                elif _is_context_ambiguous(user_query):
                    record["final_answer"] = AMBIGUOUS_ANSWER
                elif _has_conflicting_facts(user_query, documents):
                    record["final_answer"] = CONFLICT_ANSWER
                else:
                    config = load_llm_config()
                    record["llm_provider"] = config.provider
                    record["llm_model"] = config.model
                    generation_started = time.perf_counter()
                    record["final_answer"] = LLMClient(config).generate(prompt)
                    record["generation_time_ms"] = (
                        time.perf_counter() - generation_started
                    ) * 1000
        except (LLMClientError, ValueError, RuntimeError) as exc:
            record["error"] = str(exc)
            record["final_answer"] = str(exc)
        except Exception as exc:
            record["error"] = f"처리 중 예상하지 못한 오류가 발생했습니다: {exc}"
            record["final_answer"] = record["error"]
        finally:
            record["total_time_ms"] = (time.perf_counter() - started) * 1000
            self._append_log(record)
        return record

    def _append_log(self, record: dict[str, Any]) -> None:
        """프롬프트와 비밀값은 제외하고 JSON 한 줄을 누적합니다."""
        log_record = {key: value for key, value in record.items() if key != "prompt"}
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(log_record, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"로그 파일을 저장하지 못했습니다: {exc}")

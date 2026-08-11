"""세 가지 임베딩 텍스트 전략을 별도 Chroma 컬렉션에서 실제 비교합니다."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import chromadb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.chroma_store import ChromaQuestionStore
from src.embedding_model import KoreanEmbeddingModel
from src.retrieval_evaluator import evaluate_query, normalize_intent
from src.retrieval_utils import env_int, load_environment, read_and_validate_csv

COMPARISON_DIR = PROJECT_ROOT / "results" / "comparison"
ENTITY_COLUMNS = ["상품명", "가격", "수량", "크기", "시간", "날짜"]


def question_only_train(row: pd.Series) -> str:
    return row["question"]


def question_only_validation(row: pd.Series) -> str:
    return row["question"]


def intent_question_train(row: pd.Series) -> str:
    return f"인텐트: {row['intent']}\n질문: {row['question']}"


def intent_question_validation(row: pd.Series) -> str:
    return f"인텐트: {row['expected_intent']}\n질문: {row['question']}"


def question_answer_train(row: pd.Series) -> str:
    return f"고객 질문: {row['question']}\n상담 답변: {row['answer']}"


def question_answer_validation(row: pd.Series) -> str:
    # reference_answer는 실제 서비스 질의 시점에는 없는 평가용 정답이므로 상한선 분석이다.
    return f"고객 질문: {row['question']}\n상담 답변: {row['reference_answer']}"


def entity_text(row: pd.Series, question_column: str) -> str:
    """존재하고 비어 있지 않은 개체 컬럼만 결합합니다."""
    parts = [f"질문: {row[question_column]}"]
    for column in ENTITY_COLUMNS:
        value = str(row.get(column, "")).strip()
        if value:
            parts.append(f"{column}: {value}")
    return "\n".join(parts)


def evaluate_strategy(
    name: str,
    text_description: str,
    collection_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_builder: Callable[[pd.Series], str],
    validation_builder: Callable[[pd.Series], str],
    model: KoreanEmbeddingModel,
    batch_size: int,
    evaluation_depth: int,
) -> dict[str, Any]:
    """한 전략의 임베딩, 인덱싱, 검색, 저장과 지표 계산을 수행합니다."""
    print("\n" + "=" * 80)
    print(f"현재 수행 중인 실험명: {name}")
    print(f"사용 모델: {model.model_name}")
    print(f"임베딩 대상 형식: {text_description}")
    print(f"처리 문서 수: {len(train)}")
    train_texts = train.apply(train_builder, axis=1).tolist()
    validation_texts = validation.apply(validation_builder, axis=1).tolist()

    print("임베딩 진행 상황: Training 문서 임베딩 시작")
    embedding_started = time.perf_counter()
    train_embeddings = model.encode(train_texts, batch_size=batch_size)
    document_embedding_seconds = time.perf_counter() - embedding_started

    store = ChromaQuestionStore(collection_name=collection_name)
    metadatas = [
        {
            "answer": row.answer,
            "category": row.category,
            "intent": row.intent,
            "conversation_id": row.conversation_id,
            "qa_number": row.qa_number,
            "embedding_strategy": name,
        }
        for row in train.itertuples(index=False)
    ]
    store.upsert(train["document_id"].tolist(), train["question"].tolist(), train_embeddings, metadatas)
    if store.count != len(train):
        raise RuntimeError(f"{name} 컬렉션 저장 수 불일치: 기대 {len(train)}, 실제 {store.count}")

    print("평가 진행 상황: Validation 질문 임베딩 시작")
    evaluation_started = time.perf_counter()
    query_embedding_started = time.perf_counter()
    query_embeddings = model.encode(validation_texts, batch_size=batch_size)
    query_embedding_seconds = time.perf_counter() - query_embedding_started

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    print(f"평가 진행 상황: Validation {len(validation)}개 검색 시작")
    for index, row in enumerate(validation.itertuples(index=False)):
        search_started = time.perf_counter()
        results = store.query(query_embeddings[index], evaluation_depth)
        search_ms = (time.perf_counter() - search_started) * 1000
        metrics = evaluate_query(row.expected_intent, results)
        for result in results:
            metadata = result["metadata"]
            retrieved_intent = normalize_intent(metadata.get("intent"))
            detail_rows.append({
                "query_id": row.query_id, "query": row.question, "expected_intent": row.expected_intent,
                "rank": result["rank"], "retrieved_document_id": result["document_id"],
                "retrieved_question": result["question"], "retrieved_answer": metadata.get("answer", ""),
                "retrieved_intent": retrieved_intent, "distance": result["distance"],
                "intent_match": int(normalize_intent(row.expected_intent) == retrieved_intent),
            })
        summary_rows.append({
            "query_id": row.query_id, "query": row.question, "expected_intent": row.expected_intent,
            "first_matching_rank": metrics.first_matching_rank, "hit_at_1": metrics.hit_at_1,
            "hit_at_3": metrics.hit_at_3, "hit_at_5": metrics.hit_at_5,
            "reciprocal_rank": metrics.reciprocal_rank,
            "query_embedding_time_ms": query_embedding_seconds * 1000 / len(validation),
            "search_time_ms": search_ms,
        })
        if (index + 1) % 100 == 0:
            print(f"평가 진행 상황: {index + 1}/{len(validation)} 완료")

    total_evaluation_seconds = time.perf_counter() - evaluation_started
    details = pd.DataFrame(detail_rows)
    summaries = pd.DataFrame(summary_rows)
    output_dir = COMPARISON_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_dir / "retrieval_results.csv", index=False, encoding="utf-8-sig")
    summaries.to_csv(output_dir / "query_summary.csv", index=False, encoding="utf-8-sig")
    intent_metrics = summaries.groupby("expected_intent", as_index=False).agg(
        validation_query_count=("query_id", "count"),
        intent_hit_at_1=("hit_at_1", "mean"),
        intent_hit_at_3=("hit_at_3", "mean"),
        intent_hit_at_5=("hit_at_5", "mean"),
        intent_mrr=("reciprocal_rank", "mean"),
        average_search_time_ms=("search_time_ms", "mean"),
    )
    intent_metrics.to_csv(output_dir / "intent_metrics.csv", index=False, encoding="utf-8-sig")
    result = {
        "experiment_name": name,
        "embedding_text": text_description,
        "status": "completed",
        "collection_name": collection_name,
        "document_count": len(train),
        "validation_query_count": len(validation),
        "intent_hit_at_1": summaries["hit_at_1"].mean(),
        "intent_hit_at_3": summaries["hit_at_3"].mean(),
        "intent_hit_at_5": summaries["hit_at_5"].mean(),
        "intent_mrr": summaries["reciprocal_rank"].mean(),
        "hit_at_5_failure_count": int((summaries["hit_at_5"] == 0).sum()),
        "document_embedding_time_seconds": document_embedding_seconds,
        "average_document_embedding_time_ms": document_embedding_seconds * 1000 / len(train),
        "average_query_embedding_time_ms": query_embedding_seconds * 1000 / len(validation),
        "average_search_time_ms": summaries["search_time_ms"].mean(),
        "total_evaluation_time_seconds": total_evaluation_seconds,
        "skip_reason": "",
    }
    pd.DataFrame([result]).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    print(f"생성된 결과 파일: {output_dir.relative_to(PROJECT_ROOT)}")
    print(
        f"최종 비교 지표: Hit@1={result['intent_hit_at_1']:.6f}, "
        f"Hit@3={result['intent_hit_at_3']:.6f}, Hit@5={result['intent_hit_at_5']:.6f}, "
        f"MRR={result['intent_mrr']:.6f}, 실패={result['hit_at_5_failure_count']}"
    )
    return result


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for values in frame[columns].itertuples(index=False, name=None):
        formatted = []
        for value in values:
            if isinstance(value, float) and not np.isnan(value):
                formatted.append(f"{value:.6f}")
            else:
                formatted.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join([header, separator, *rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="새 비교 산출물을 의도적으로 재생성")
    args = parser.parse_args()
    load_environment()
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = COMPARISON_DIR / "super_embedding_strategy_comparison.csv"
    document_path = PROJECT_ROOT / "docs/embedding_strategy_comparison.md"
    if (comparison_path.exists() or document_path.exists()) and not args.force:
        raise FileExistsError("기존 비교 결과가 있습니다. 확인 후 --force를 사용하세요.")

    train_required = ["document_id", "category", "conversation_id", "qa_number", "intent", "question", "answer"]
    validation_required = ["query_id", "question", "reference_answer", "expected_intent", "category", "conversation_id", "qa_number"]
    train = read_and_validate_csv(PROJECT_ROOT / "data/experiment/super_train_sample_5000.csv", train_required)
    validation = read_and_validate_csv(PROJECT_ROOT / "data/experiment/super_validation_sample_500.csv", validation_required)
    print(f"Training 실제 컬럼: {list(train.columns)}")
    print(f"Validation 실제 컬럼: {list(validation.columns)}")

    collection_names = {
        "baseline_question_only": "super_baseline_question_only",
        "intent_question": "super_intent_question",
        "question_answer": "super_question_answer",
        "question_entities": "super_question_entities",
    }
    client = chromadb.PersistentClient(path=str(PROJECT_ROOT / os.getenv("CHROMA_PATH", "./chroma_db")))
    existing_collections = {collection.name for collection in client.list_collections()}
    collisions = [name for name in collection_names.values() if name in existing_collections]
    if collisions and not args.force:
        raise FileExistsError(f"새 실험 컬렉션이 이미 있습니다. 기존 baseline은 건드리지 않습니다: {collisions}")

    batch_size = env_int("EMBEDDING_BATCH_SIZE", 64)
    evaluation_depth = max(env_int("TOP_K", 5), env_int("EVALUATION_DEPTH", 20))
    model = KoreanEmbeddingModel()
    print(f"사용 모델: {model.model_name}, 차원: {model.dimension}")
    strategies = [
        ("baseline_question_only", "question", question_only_train, question_only_validation),
        ("intent_question", "인텐트: {intent} + 질문: {question}", intent_question_train, intent_question_validation),
        ("question_answer", "고객 질문: {question} + 상담 답변: {answer}", question_answer_train, question_answer_validation),
    ]
    results = []
    for name, description, train_builder, validation_builder in strategies:
        results.append(evaluate_strategy(
            name, description, collection_names[name], train, validation,
            train_builder, validation_builder, model, batch_size, evaluation_depth,
        ))

    missing_train_entities = [column for column in ENTITY_COLUMNS if column not in train.columns]
    missing_validation_entities = [column for column in ENTITY_COLUMNS if column not in validation.columns]
    if not missing_train_entities and not missing_validation_entities:
        results.append(evaluate_strategy(
            "question_entities", "질문 + 값이 있는 개체 정보", collection_names["question_entities"], train, validation,
            lambda row: entity_text(row, "question"), lambda row: entity_text(row, "question"),
            model, batch_size, evaluation_depth,
        ))
    else:
        reason = f"표본에 원본 개체 컬럼이 없음: Training 누락={missing_train_entities}, Validation 누락={missing_validation_entities}"
        print(f"현재 수행 중인 실험명: question_entities - 건너뜀 ({reason})")
        results.append({
            "experiment_name": "question_entities", "embedding_text": "질문 + 값이 있는 개체 정보",
            "status": "skipped", "collection_name": collection_names["question_entities"], "document_count": len(train),
            "validation_query_count": len(validation), "intent_hit_at_1": np.nan, "intent_hit_at_3": np.nan,
            "intent_hit_at_5": np.nan, "intent_mrr": np.nan, "hit_at_5_failure_count": np.nan,
            "document_embedding_time_seconds": np.nan, "average_document_embedding_time_ms": np.nan,
            "average_query_embedding_time_ms": np.nan, "average_search_time_ms": np.nan,
            "total_evaluation_time_seconds": np.nan, "skip_reason": reason,
        })

    comparison = pd.DataFrame(results)
    baseline_original = pd.read_csv(PROJECT_ROOT / "results/retrieval/super_retrieval_metrics.csv", encoding="utf-8-sig").iloc[0]
    reproduced = comparison.loc[comparison["experiment_name"] == "baseline_question_only"].iloc[0]
    comparison["baseline_original_match"] = ""
    comparison.loc[comparison["experiment_name"] == "baseline_question_only", "baseline_original_match"] = str(all(
        np.isclose(reproduced[key], baseline_original[key], atol=1e-12)
        for key in ("intent_hit_at_1", "intent_hit_at_3", "intent_hit_at_5", "intent_mrr")
    ))
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    completed = comparison.loc[comparison["status"] == "completed"].copy()
    best_deployable = completed.loc[completed["experiment_name"] == "baseline_question_only"].iloc[0]
    intent_result = completed.loc[completed["experiment_name"] == "intent_question"].iloc[0]
    answer_result = completed.loc[completed["experiment_name"] == "question_answer"].iloc[0]
    recommendations = [
        "사람이 관련 Training 문서를 직접 지정한 평가셋 구축: 현재 Intent Hit은 같은 인텐트 안의 답변 정확성을 판단하지 못한다.",
    ]
    if intent_result["intent_hit_at_5"] > best_deployable["intent_hit_at_5"]:
        recommendations.append("인텐트 분류 후 제한 검색: 인텐트+질문 상한선 결과를 실제 서비스 입력으로 바꾸려면 독립적인 인텐트 분류 성능을 먼저 검증해야 한다.")
    recommendations.append("다른 한국어 임베딩 모델 비교: 질문-only 방식의 표현 차이와 세부 인텐트 혼동을 데이터 누수 없이 개선할 수 있는지 확인한다.")
    recommendations = recommendations[:3]

    table_columns = [
        "experiment_name", "embedding_text", "status", "intent_hit_at_1", "intent_hit_at_3",
        "intent_hit_at_5", "intent_mrr", "hit_at_5_failure_count", "document_embedding_time_seconds",
        "average_search_time_ms",
    ]
    document = f"""# 임베딩 전략 비교

## 실험 목적

같은 모델과 Training 5,000개·Validation 500개를 사용하고 임베딩 입력 텍스트만 바꿨다. 각 실험은 별도 Chroma cosine 거리 컬렉션을 사용하며 기존 `super_questions` baseline 컬렉션과 결과 파일은 수정하지 않았다.

## 실제 측정 비교

{markdown_table(comparison, table_columns)}

질문-only 재현 지표와 기존 baseline 지표의 완전 일치 여부: **{comparison.loc[comparison['experiment_name'] == 'baseline_question_only', 'baseline_original_match'].iloc[0]}**

## 방식별 목적과 해석

### baseline_question_only

실제 사용자가 제공하는 질문만으로 검색하는 서비스 적용 가능한 비교 기준이다. 질문에 없는 답변이나 정답 인텐트를 사용하지 않는다.

### intent_question

Training에는 `intent`, Validation에는 정답 `expected_intent`를 입력 문자열에 직접 넣었다. 현재 평가 지표도 인텐트 일치이므로 **라벨 누수로 유리해질 수 있는 분석용 상한선 실험**이다. 실제 서비스에서는 사용자 질문의 정답 인텐트를 미리 알 수 없으므로 별도 인텐트 분류 모델과 그 오류를 포함한 종단 평가가 필요하다. 최종 운영 방식으로 단정할 수 없다.

### question_answer

Training 질문과 답변, Validation 질문과 평가용 `reference_answer`를 함께 임베딩했다. 답변 의미가 문서 구분에 도움이 되는지 보는 상한선이지만, 실제 검색 시 사용자는 정답 답변을 제공하지 않으므로 그대로 서비스에 적용할 수 없다. 짧은 답변, 동일 질문의 서로 다른 답변, 상품별 가격·수량 표현이 질문 의미를 흐리는 문제도 있다.

### question_entities

상태: **{comparison.loc[comparison['experiment_name'] == 'question_entities', 'status'].iloc[0]}**. 이유: {comparison.loc[comparison['experiment_name'] == 'question_entities', 'skip_reason'].iloc[0]}. 원본 개체 라벨을 임의로 복원하지 않았다. 이후 원본 발화에서 개체 컬럼을 QA 단위로 집계한 데이터가 준비되면 실제 서비스에서 얻을 수 있는 개체만 사용해 재실험해야 한다.

## 속도와 실제 적용 가능성

문서 임베딩 시간은 5,000개 전체를 각 형식으로 다시 임베딩한 실제 시간이다. 평균 검색 시간은 Chroma가 반환한 원래 거리로 20개 후보를 검색한 시간이다. 텍스트가 길어지면 임베딩 비용이 증가할 수 있으며, 수치가 높더라도 정답 라벨이나 정답 답변을 입력한 전략은 서비스 성능으로 해석하면 안 된다.

질문+답변이 높은 지표를 보이더라도 같은 질문의 상이한 답변 문제를 해결했다고 단정할 수 없다. Intent Hit은 답변 정확성을 측정하지 않으며 특정 인텐트만 개선됐을 수도 있어 각 폴더의 `query_summary.csv`를 인텐트별로 추가 확인해야 한다.

## 최종 추천

현재 운영 가능한 기본 방식은 **question-only**다. intent+question과 question+answer는 누수가 있는 상한선 분석으로만 사용한다. 단순히 가장 높은 수치를 최종 방식으로 선택하지 않는다.

우선순위가 높은 다음 실험은 다음과 같다.

{chr(10).join(f'{index + 1}. {text}' for index, text in enumerate(recommendations))}

Training 10,000개 확대는 표본 수와 Hit@5의 상관이 약하거나 음수였던 현재 실패 분석만으로 최우선이라고 보기 어렵다. 먼저 평가 라벨과 모델·인텐트 혼동을 개선한 뒤 확대 효과를 측정하는 편이 낫다.

## 실행 방법

```powershell
C:\\Python312\\python.exe scripts\\compare_embedding_strategies.py
```

새 비교 결과와 컬렉션을 확인하고 의도적으로 재실행할 때만 `--force`를 사용한다. LLM, 외부 생성 API와 LangChain은 사용하지 않는다. 별도 가상환경 사용을 권장하며, 현재 전역 Python에는 기존 Google 패키지와 `protobuf` 충돌 경고가 있었다.
"""
    document_path.write_text(document, encoding="utf-8-sig")
    print(f"\n생성된 결과 파일: {comparison_path.relative_to(PROJECT_ROOT)}")
    print(f"생성된 결과 파일: {document_path.relative_to(PROJECT_ROOT)}")
    print("\n최종 비교 지표:")
    print(comparison[table_columns].to_string(index=False))


if __name__ == "__main__":
    main()

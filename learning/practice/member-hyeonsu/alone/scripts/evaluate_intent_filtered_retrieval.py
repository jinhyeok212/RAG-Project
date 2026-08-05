"""예측 인텐트만 사용하여 다섯 Chroma 검색 전략을 실제 비교합니다."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT))

from src.embedding_model import KoreanEmbeddingModel
from src.intent_filtered_retrieval import IntentFilteredRetriever
from src.retrieval_evaluator import evaluate_query
from src.retrieval_utils import env_int, load_environment

VALIDATION_PATH = ROOT / "data/experiment/super_validation_sample_500.csv"
PREDICTION_PATH = ROOT / "results/intent_classifier/super_intent_classifier_predictions.csv"
CLASSIFIER_METRICS_PATH = ROOT / "results/intent_classifier/super_intent_classifier_metrics.csv"
MODEL_PATH = ROOT / "models/super_intent_classifier.joblib"
RESULT_DIR = ROOT / "results/intent_filtered_retrieval"
DOCUMENT_PATH = ROOT / "docs/intent_filtered_retrieval_experiment.md"
STRATEGIES = ["full", "top1_strict", "top1_fallback", "top3_strict", "top3_fallback"]


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    selected = frame[columns] if limit is None else frame[columns].head(limit)
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for values in selected.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            if isinstance(value, float) and not np.isnan(value):
                formatted.append(f"{value:.6f}")
            else:
                formatted.append(str(value).replace("|", "\\|").replace("\n", " ")[:180])
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join([header, separator, *rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="새 제한검색 결과를 의도적으로 교체")
    args = parser.parse_args()
    targets = [
        RESULT_DIR / "super_intent_filtered_retrieval_comparison.csv",
        RESULT_DIR / "super_intent_filtered_retrieval_results.csv",
        RESULT_DIR / "super_intent_filtered_query_metrics.csv",
        RESULT_DIR / "super_intent_filtered_failures.csv",
        RESULT_DIR / "super_intent_filtered_intent_metrics.csv",
        DOCUMENT_PATH,
    ]
    existing = [str(path) for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError("기존 결과가 있어 중단합니다. 확인 후 --force를 사용하세요:\n" + "\n".join(existing))
    if not MODEL_PATH.exists() or not PREDICTION_PATH.exists():
        raise FileNotFoundError("먼저 train_intent_classifier.py를 실행하세요.")

    load_environment()
    validation = pd.read_csv(VALIDATION_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    predictions = pd.read_csv(PREDICTION_PATH, encoding="utf-8-sig", dtype={"query_id": str})
    required_validation = ["query_id", "question", "expected_intent"]
    required_prediction = [
        "query_id", "predicted_intent_top1", "predicted_probability_top1",
        "predicted_intent_top2", "predicted_probability_top2",
        "predicted_intent_top3", "predicted_probability_top3",
    ]
    for label, frame, columns in (("Validation", validation, required_validation), ("예측", predictions, required_prediction)):
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{label} 파일에 컬럼이 없습니다: {missing}")
    data = validation.merge(predictions.drop(columns=["question", "true_intent"], errors="ignore"), on="query_id", how="inner", validate="one_to_one")
    if len(data) != 500 or len(validation) != 500 or len(predictions) != 500:
        raise ValueError(f"모든 전략은 동일한 Validation 500개를 사용해야 합니다: {len(validation)}, {len(predictions)}, {len(data)}")

    # 저장 모델도 실제로 로드해 query 순서의 예측이 결과 CSV와 같은지 다시 확인합니다.
    classifier = joblib.load(MODEL_PATH)
    loaded_predictions = classifier.predict(data["question"])
    if not np.array_equal(loaded_predictions, data["predicted_intent_top1"].to_numpy()):
        raise RuntimeError("저장 모델 예측과 예측 CSV의 Top-1이 다릅니다.")

    top_k = 5
    evaluation_depth = max(20, env_int("EVALUATION_DEPTH", 20))
    model = KoreanEmbeddingModel()
    print(f"질문 임베딩 모델: {model.model_name}")
    embed_started = time.perf_counter()
    embeddings = model.encode(data["question"].tolist(), batch_size=env_int("EMBEDDING_BATCH_SIZE", 64))
    embedding_seconds = time.perf_counter() - embed_started
    print(f"Validation 500개 임베딩 완료: {embedding_seconds:.3f}초")

    retriever = IntentFilteredRetriever(collection_name="super_questions")
    detail_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        print(f"검색 전략 평가 중: {strategy}")
        for index, row in data.iterrows():
            # 검색기에는 예측 인텐트만 넘깁니다. expected_intent는 아래 평가에서만 사용합니다.
            predicted_intents = [
                str(row["predicted_intent_top1"]),
                str(row["predicted_intent_top2"]),
                str(row["predicted_intent_top3"]),
            ]
            displayed, evaluated, search_ms = retriever.search(
                embeddings[index], strategy, predicted_intents, top_k=top_k,
                evaluation_depth=evaluation_depth,
            )
            eval_rows = [
                {"rank": rank + 1, "metadata": {"intent": item["intent"]}}
                for rank, item in enumerate(evaluated)
            ]
            metrics = evaluate_query(str(row["expected_intent"]), eval_rows)
            query_rows.append({
                "query_id": row["query_id"], "question": row["question"], "true_intent": row["expected_intent"],
                "predicted_intent_top1": row["predicted_intent_top1"], "predicted_probability_top1": row["predicted_probability_top1"],
                "predicted_intent_top2": row["predicted_intent_top2"], "predicted_probability_top2": row["predicted_probability_top2"],
                "predicted_intent_top3": row["predicted_intent_top3"], "predicted_probability_top3": row["predicted_probability_top3"],
                "strategy": strategy, "first_matching_rank": metrics.first_matching_rank,
                "hit_at_1": metrics.hit_at_1, "hit_at_3": metrics.hit_at_3, "hit_at_5": metrics.hit_at_5,
                "reciprocal_rank": metrics.reciprocal_rank, "search_time_ms": search_ms,
                "returned_document_count": len(displayed),
            })
            for rank, item in enumerate(displayed, start=1):
                detail_rows.append({
                    "query_id": row["query_id"], "question": row["question"], "true_intent": row["expected_intent"],
                    "predicted_intent_top1": row["predicted_intent_top1"], "predicted_probability_top1": row["predicted_probability_top1"],
                    "predicted_intent_top2": row["predicted_intent_top2"], "predicted_probability_top2": row["predicted_probability_top2"],
                    "predicted_intent_top3": row["predicted_intent_top3"], "predicted_probability_top3": row["predicted_probability_top3"],
                    "strategy": strategy, "rank": rank, "retrieved_document_id": item["document_id"],
                    "retrieved_question": item["question"], "retrieved_answer": item["answer"],
                    "retrieved_intent": item["intent"], "distance": item["distance"],
                    "intent_match": int(item["intent"] == row["expected_intent"]),
                })
        print(f"{strategy}: 500/500 완료")

    details = pd.DataFrame(detail_rows)
    queries = pd.DataFrame(query_rows)
    comparison_rows = []
    for strategy, group in queries.groupby("strategy", sort=False):
        comparison_rows.append({
            "strategy": strategy,
            "intent_hit_at_1": group["hit_at_1"].mean(),
            "intent_hit_at_3": group["hit_at_3"].mean(),
            "intent_hit_at_5": group["hit_at_5"].mean(),
            "intent_mrr": group["reciprocal_rank"].mean(),
            "hit_at_5_failure_count": int((group["hit_at_5"] == 0).sum()),
            "average_search_time_ms": group["search_time_ms"].mean(),
            "p50_search_time_ms": group["search_time_ms"].quantile(0.50),
            "p95_search_time_ms": group["search_time_ms"].quantile(0.95),
            "average_returned_documents": group["returned_document_count"].mean(),
        })
    comparison = pd.DataFrame(comparison_rows)

    intent_metrics = queries.groupby(["strategy", "true_intent"], as_index=False).agg(
        validation_query_count=("query_id", "count"), intent_hit_at_1=("hit_at_1", "mean"),
        intent_hit_at_3=("hit_at_3", "mean"), intent_hit_at_5=("hit_at_5", "mean"),
        intent_mrr=("reciprocal_rank", "mean"), average_search_time_ms=("search_time_ms", "mean"),
    )

    pivot = queries.pivot(index="query_id", columns="strategy", values="hit_at_5")
    base = data.set_index("query_id")
    probability_margin = base["predicted_probability_top1"].astype(float) - base["predicted_probability_top2"].astype(float)
    low_probability_threshold = float(base["predicted_probability_top1"].astype(float).quantile(0.25))
    low_margin_threshold = float(probability_margin.quantile(0.25))
    cases: list[dict[str, Any]] = []

    def add_cases(mask: pd.Series, case_type: str, reason: str) -> None:
        for query_id in mask[mask].index:
            row = base.loc[query_id]
            top1_detail = details[(details["query_id"] == query_id) & (details["strategy"] == "top1_strict") & (details["rank"] == 1)]
            top1_question = "" if top1_detail.empty else top1_detail.iloc[0]["retrieved_question"]
            cases.append({
                "failure_case_type": case_type, "query_id": query_id, "question": row["question"],
                "true_intent": row["expected_intent"], "predicted_intent_top1": row["predicted_intent_top1"],
                "predicted_probability_top1": row["predicted_probability_top1"],
                "predicted_intent_top2": row["predicted_intent_top2"],
                "predicted_probability_top2": row["predicted_probability_top2"],
                "probability_margin_top1_top2": float(row["predicted_probability_top1"]) - float(row["predicted_probability_top2"]),
                "full_hit_at_5": int(pivot.loc[query_id, "full"]),
                "top1_strict_hit_at_5": int(pivot.loc[query_id, "top1_strict"]),
                "top3_strict_hit_at_5": int(pivot.loc[query_id, "top3_strict"]),
                "top1_retrieved_question": top1_question, "failure_reason_candidate": reason,
                "manual_review_note": "",
            })

    top1_correct = base["predicted_intent_top1"] == base["expected_intent"]
    add_cases(top1_correct & (pivot["top1_strict"] == 0), "Top1 예측 정답·제한검색 실패", "분류는 맞았지만 제한 후보 안에서 기대 인텐트 문서가 Top-5에 없음")
    add_cases(~top1_correct & (pivot["top1_strict"] == 0), "Top1 예측 오답·제한검색 실패", "잘못 예측한 인텐트로 검색 범위가 제한됨")
    add_cases((pivot["top1_strict"] == 0) & (pivot["top3_strict"] == 1), "Top1 실패·Top3 성공", "정답 인텐트가 예측 상위 3개 안에는 포함될 가능성")
    add_cases((pivot["full"] == 1) & (pivot["top1_strict"] == 0), "Full 성공·Top1 제한 실패", "전체 검색은 성공했지만 Top-1 필터가 관련 문서를 제거함")
    add_cases((pivot["full"] == 0) & ((pivot["top1_strict"] == 1) | (pivot["top3_strict"] == 1)), "Full 실패·제한검색 성공", "예측 인텐트 제한이 전체 검색의 오검색 후보를 제거함")
    uncertainty = (base["predicted_probability_top1"].astype(float) <= low_probability_threshold) | (probability_margin <= low_margin_threshold)
    add_cases(uncertainty, "낮은 확률 또는 근접 확률", f"Top1 확률 하위 25%≤{low_probability_threshold:.6f} 또는 Top1-Top2 차이 하위 25%≤{low_margin_threshold:.6f}")
    failures = pd.DataFrame(cases)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(targets[0], index=False, encoding="utf-8-sig")
    details.to_csv(targets[1], index=False, encoding="utf-8-sig")
    queries.to_csv(targets[2], index=False, encoding="utf-8-sig")
    failures.to_csv(targets[3], index=False, encoding="utf-8-sig")
    intent_metrics.to_csv(targets[4], index=False, encoding="utf-8-sig")

    classifier_metrics = pd.read_csv(CLASSIFIER_METRICS_PATH, encoding="utf-8-sig")
    final_classifier = classifier_metrics[classifier_metrics["record_type"] == "validation_final"].iloc[0]
    failure_counts = failures["failure_case_type"].value_counts().rename_axis("failure_case_type").reset_index(name="case_count")
    best = comparison.sort_values(["intent_hit_at_5", "intent_mrr", "average_search_time_ms"], ascending=[False, False, True]).iloc[0]
    baseline = comparison[comparison["strategy"] == "full"].iloc[0]
    document = f"""# 예측 인텐트 제한 Retrieval 실험

## 왜 인텐트 분류기가 필요한가?

Training에는 정답 `intent`가 있지만 실제 사용자의 새 질문에는 라벨이 없다. 정답 인텐트를 검색 필터에 직접 넣으면 평가 정답을 미리 사용한 데이터 누수다. 이 실험은 Training 질문과 기존 인텐트 이름만 학습한 분류기가 새 질문의 인텐트를 예측하고, 검색기에는 그 **예측값만** 전달한다. Validation의 `expected_intent`는 검색이 끝난 뒤 지표 계산에만 사용했다.

## 분류기

TF-IDF와 Logistic Regression을 sklearn Pipeline으로 구성했다. Training 내부 홀드아웃에서 char n-gram balanced/unbalanced와 word n-gram balanced를 비교하고 Macro F1이 가장 높은 설정을 선택한 뒤 전체 5,000개로 재학습했다. Validation은 최종 평가에만 사용했다. 샘플 1개짜리 인텐트도 삭제하지 않았으며 내부 홀드아웃에는 넣지 않고 학습에 남겼다.

| 지표 | 실제 값 |
|---|---:|
| 선택 설정 | {final_classifier['configuration']} |
| Accuracy / Top-1 | {final_classifier['accuracy']:.6f} |
| Macro F1 | {final_classifier['macro_f1']:.6f} |
| Weighted F1 | {final_classifier['weighted_f1']:.6f} |
| Top-3 Accuracy | {final_classifier['top3_accuracy']:.6f} |
| Top-5 Accuracy | {final_classifier['top5_accuracy']:.6f} |
| 저장 모델 재로드 동일 | {final_classifier['reload_prediction_identical']} |

## 검색 전략

- `full`: 기존 5,000개 전체 검색
- `top1_strict`: 예측 Top-1 인텐트만 검색
- `top1_fallback`: Top-1 제한 결과가 5개 미만일 때 전체 검색으로 보충
- `top3_strict`: 예측 Top-3 인텐트 중에서 검색
- `top3_fallback`: Top-3 제한 결과가 5개 미만일 때 전체 검색으로 보충

Strict는 분류 오류에 민감하다. Fallback은 희소 인텐트에서 문서 수 부족을 보완하지만 잘못된 전체 검색 문서가 다시 들어올 수 있다. 모든 결과는 document_id 중복 제거 후 Chroma 원래 distance 오름차순으로 정렬했다. distance는 유사도가 아니며 낮을수록 가깝다.

## 실제 비교

{markdown_table(comparison, list(comparison.columns))}

기존 참고 baseline은 Hit@1=0.452, Hit@3=0.604, Hit@5=0.676, MRR=0.551851이다. 위 `full`은 기존 컬렉션에서 동일 500개를 다시 검색한 실제 값이며 결과가 낮더라도 그대로 기록했다. MRR은 기존 실험과 맞춰 최대 20위에서 첫 인텐트 일치를 확인했고 상세 CSV에는 최종 Top-5만 저장했다.

실측 기준 가장 높은 Hit@5 전략은 **{best['strategy']}**이다. 다만 단일 수치만으로 운영 전략을 확정하지 말고 분류 오류 시 실패, 지연, 반환 문서 수와 fallback 동작을 함께 봐야 한다. 기본 RAG는 요청대로 여전히 `full`이며 자동 변경하지 않았다.

## 실패 사례 유형

{markdown_table(failure_counts, list(failure_counts.columns))}

낮은 확률은 임의의 고정 임계값이 아니라 이번 500개 분포의 하위 25%를 사용했다. Top-1 확률 기준은 {low_probability_threshold:.6f}, Top1-Top2 확률 차이 기준은 {low_margin_threshold:.6f}이다. 유형은 서로 겹칠 수 있어 합계가 500을 넘을 수 있으며 자동 원인 확정이 아니다.

## 누수·정합성 검증

- Training 5,000개만 분류 학습에 사용하고 Validation 500개는 최종 평가에만 사용했다.
- 검색 함수에는 `predicted_intent_top1~3`만 전달했다. `expected_intent`는 검색 결과 반환 후 Hit/MRR 계산에만 사용했다.
- 저장 Pipeline 재로드 후 500개 예측 확률과 Top-1이 동일했다.
- 모든 전략은 같은 500개 질문과 같은 임베딩을 사용했다.
- 상세 결과는 query·strategy별 최대 5개이며 document_id 중복이 없고 distance 오름차순이다.
- 기존 `super_questions`는 읽기 전용 `get_collection/query`만 사용했다.

## 현재 한계와 다음 개선

클래스가 199개로 많고 Training 문서가 1~2개뿐인 희소 인텐트가 있다. Validation도 인텐트별 질문 수가 적어 클래스별 지표 변동이 크다. TF-IDF는 표현이 크게 달라지거나 문맥이 필요한 질문에 약하고, 인텐트 라벨 자체가 질문/요청/확인처럼 세분화되어 혼동될 수 있다.

다음 단계에서는 낮은 확률에서 자동으로 `full` 또는 Top-3로 전환하는 confidence-aware 전략, 인텐트별 최소 데이터 보강, 사람이 지정한 관련 문서 평가셋을 우선 검토할 수 있다.

## 실행 방법

```powershell
C:\\Python312\\python.exe scripts\\train_intent_classifier.py
C:\\Python312\\python.exe scripts\\evaluate_intent_filtered_retrieval.py
```

의도적으로 새 결과를 교체할 때만 두 명령에 `--force`를 추가한다. 기본 RAG/LLM 코드는 변경하지 않았다.
"""
    DOCUMENT_PATH.write_text(document, encoding="utf-8-sig")
    print("\n검색 전략 비교")
    print(comparison.to_string(index=False))
    for target in targets:
        print(f"생성 파일: {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

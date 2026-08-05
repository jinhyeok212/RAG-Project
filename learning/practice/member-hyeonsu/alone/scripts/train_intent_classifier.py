"""Training 질문으로 TF-IDF + Logistic Regression 인텐트 분류기를 학습합니다."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/experiment/super_train_sample_5000.csv"
VALIDATION_PATH = ROOT / "data/experiment/super_validation_sample_500.csv"
MODEL_PATH = ROOT / "models/super_intent_classifier.joblib"
RESULT_DIR = ROOT / "results/intent_classifier"


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label}에 필수 컬럼이 없습니다: {missing}")


def make_internal_split(labels: pd.Series, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """모든 클래스를 학습에 남기면서 샘플이 2개 이상인 클래스만 일부 검증에 배정합니다."""
    random = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for _, indices in labels.groupby(labels).groups.items():
        values = np.asarray(list(indices), dtype=int)
        random.shuffle(values)
        holdout = 0 if len(values) == 1 else max(1, int(round(len(values) * 0.2)))
        holdout = min(holdout, len(values) - 1)
        validation_indices.extend(values[:holdout].tolist())
        train_indices.extend(values[holdout:].tolist())
    return np.asarray(sorted(train_indices)), np.asarray(sorted(validation_indices))


def make_pipeline(analyzer: str, class_weight: str | None) -> Pipeline:
    """짧은 한국어에는 char n-gram을 우선하고 word 설정도 비교할 수 있게 구성합니다."""
    if analyzer == "char":
        vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 5), min_df=1, sublinear_tf=True,
            max_features=60_000, dtype=np.float32,
        )
    else:
        vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
            max_features=40_000, dtype=np.float32,
        )
    classifier = LogisticRegression(
        class_weight=class_weight, max_iter=1_500, random_state=42,
        solver="lbfgs", n_jobs=None,
    )
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def top_k_accuracy(probabilities: np.ndarray, classes: np.ndarray, truth: pd.Series, k: int) -> float:
    """정답 인텐트가 확률 상위 k개 안에 있는 질문 비율을 계산합니다."""
    k = min(k, probabilities.shape[1])
    top_indices = np.argpartition(-probabilities, kth=k - 1, axis=1)[:, :k]
    top_labels = classes[top_indices]
    return float(np.mean([value in labels for value, labels in zip(truth, top_labels)]))


def metric_row(
    name: str, split: str, truth: pd.Series, predictions: np.ndarray,
    probabilities: np.ndarray, classes: np.ndarray, elapsed: float,
) -> dict[str, Any]:
    return {
        "record_type": split,
        "configuration": name,
        "accuracy": accuracy_score(truth, predictions),
        "macro_f1": f1_score(truth, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(truth, predictions, average="weighted", zero_division=0),
        "top1_accuracy": top_k_accuracy(probabilities, classes, truth, 1),
        "top3_accuracy": top_k_accuracy(probabilities, classes, truth, 3),
        "top5_accuracy": top_k_accuracy(probabilities, classes, truth, 5),
        "question_count": len(truth),
        "training_or_fit_time_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="기존 모델과 새 분류 결과를 의도적으로 교체")
    args = parser.parse_args()
    targets = [
        MODEL_PATH,
        RESULT_DIR / "super_intent_classifier_metrics.csv",
        RESULT_DIR / "super_intent_classifier_predictions.csv",
        RESULT_DIR / "super_intent_classification_report.csv",
        RESULT_DIR / "super_intent_confusion_matrix.csv",
        RESULT_DIR / "super_intent_confusion_pairs.csv",
        RESULT_DIR / "super_intent_class_distribution.csv",
    ]
    existing = [str(path) for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError("기존 결과가 있어 중단합니다. 확인 후 --force를 사용하세요:\n" + "\n".join(existing))

    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    validation = pd.read_csv(VALIDATION_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    require_columns(train, ["document_id", "question", "answer", "intent"], "Training")
    require_columns(validation, ["query_id", "question", "reference_answer", "expected_intent"], "Validation")
    for frame, columns in ((train, ["question", "intent"]), (validation, ["question", "expected_intent"])):
        for column in columns:
            frame[column] = frame[column].astype(str).str.strip()
            if frame[column].eq("").any():
                raise ValueError(f"{column}에 빈 값이 있습니다.")
    if set(validation["query_id"]) & set(train["document_id"]):
        raise ValueError("Training document_id와 Validation query_id가 겹칩니다. 누수 여부를 확인하세요.")

    print(f"Training 실제 컬럼: {list(train.columns)}")
    print(f"Validation 실제 컬럼: {list(validation.columns)}")
    print(f"Training: {len(train)}개, Validation: {len(validation)}개, 인텐트: {train['intent'].nunique()}개")

    internal_train, internal_validation = make_internal_split(train["intent"])
    candidates = [
        ("char_balanced", "char", "balanced"),
        ("char_unbalanced", "char", None),
        ("word_balanced", "word", "balanced"),
    ]
    candidate_rows = []
    best: tuple[float, str, str | None] | None = None
    for name, analyzer, class_weight in candidates:
        print(f"내부 비교 학습: {name} (analyzer={analyzer}, class_weight={class_weight})")
        pipeline = make_pipeline(analyzer, class_weight)
        started = time.perf_counter()
        pipeline.fit(train.loc[internal_train, "question"], train.loc[internal_train, "intent"])
        elapsed = time.perf_counter() - started
        probabilities = pipeline.predict_proba(train.loc[internal_validation, "question"])
        predictions = pipeline.classes_[np.argmax(probabilities, axis=1)]
        row = metric_row(
            name, "training_internal_holdout", train.loc[internal_validation, "intent"],
            predictions, probabilities, pipeline.classes_, elapsed,
        )
        candidate_rows.append(row)
        score = (row["macro_f1"], name, class_weight)
        if best is None or score[0] > best[0]:
            best = score
    assert best is not None
    selected_name = best[1]
    selected_analyzer = next(analyzer for name, analyzer, _ in candidates if name == selected_name)
    selected_weight = next(weight for name, _, weight in candidates if name == selected_name)
    print(f"선택 설정: {selected_name}")

    final_model = make_pipeline(selected_analyzer, selected_weight)
    final_started = time.perf_counter()
    final_model.fit(train["question"], train["intent"])
    final_fit_seconds = time.perf_counter() - final_started
    probabilities = final_model.predict_proba(validation["question"])
    ordered = np.argsort(-probabilities, axis=1)
    predictions = final_model.classes_[ordered[:, 0]]
    final_row = metric_row(
        selected_name, "validation_final", validation["expected_intent"], predictions,
        probabilities, final_model.classes_, final_fit_seconds,
    )
    candidate_rows.append(final_row)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    reloaded: Pipeline = joblib.load(MODEL_PATH)
    reloaded_probabilities = reloaded.predict_proba(validation["question"])
    reload_equal = bool(np.array_equal(reloaded.predict(validation["question"]), predictions) and np.allclose(reloaded_probabilities, probabilities, atol=0, rtol=0))
    if not reload_equal:
        raise RuntimeError("저장 후 재로드한 Pipeline의 예측이 달라졌습니다.")
    for row in candidate_rows:
        row["selected_final_configuration"] = selected_name
        row["reload_prediction_identical"] = reload_equal
    pd.DataFrame(candidate_rows).to_csv(targets[1], index=False, encoding="utf-8-sig")

    prediction_rows = []
    for index, row in validation.iterrows():
        labels = final_model.classes_[ordered[index, :5]]
        probs = probabilities[index, ordered[index, :5]]
        prediction_rows.append({
            "query_id": row["query_id"], "question": row["question"], "true_intent": row["expected_intent"],
            "predicted_intent_top1": labels[0], "predicted_probability_top1": probs[0],
            "predicted_intent_top2": labels[1], "predicted_probability_top2": probs[1],
            "predicted_intent_top3": labels[2], "predicted_probability_top3": probs[2],
            "predicted_intent_top4": labels[3], "predicted_probability_top4": probs[3],
            "predicted_intent_top5": labels[4], "predicted_probability_top5": probs[4],
            "top1_correct": int(row["expected_intent"] == labels[0]),
            "top3_correct": int(row["expected_intent"] in labels[:3]),
            "top5_correct": int(row["expected_intent"] in labels[:5]),
        })
    pd.DataFrame(prediction_rows).to_csv(targets[2], index=False, encoding="utf-8-sig")

    labels = list(final_model.classes_)
    report = classification_report(
        validation["expected_intent"], predictions, labels=labels,
        output_dict=True, zero_division=0,
    )
    report_rows = []
    for label in labels:
        values = report[label]
        report_rows.append({"intent": label, "precision": values["precision"], "recall": values["recall"], "f1": values["f1-score"], "support": int(values["support"])})
    pd.DataFrame(report_rows).to_csv(targets[3], index=False, encoding="utf-8-sig")

    matrix = confusion_matrix(validation["expected_intent"], predictions, labels=labels)
    matrix_frame = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix_frame.index.name = "true_intent"
    matrix_frame.to_csv(targets[4], encoding="utf-8-sig")
    pair_rows = []
    for true_index, true_label in enumerate(labels):
        for predicted_index, predicted_label in enumerate(labels):
            count = int(matrix[true_index, predicted_index])
            if true_label != predicted_label and count:
                pair_rows.append({"true_intent": true_label, "predicted_intent": predicted_label, "confusion_count": count})
    pd.DataFrame(pair_rows).sort_values(["confusion_count", "true_intent", "predicted_intent"], ascending=[False, True, True]).to_csv(targets[5], index=False, encoding="utf-8-sig")

    train_counts = train["intent"].value_counts()
    validation_counts = validation["expected_intent"].value_counts()
    distribution = pd.DataFrame({
        "intent": labels,
        "training_count": [int(train_counts.get(label, 0)) for label in labels],
        "validation_count": [int(validation_counts.get(label, 0)) for label in labels],
    })
    distribution["is_very_rare_training_class"] = distribution["training_count"] <= 2
    distribution.to_csv(targets[6], index=False, encoding="utf-8-sig")

    print("\n최종 Validation 분류 성능")
    for key in ("accuracy", "macro_f1", "weighted_f1", "top1_accuracy", "top3_accuracy", "top5_accuracy"):
        print(f"{key}: {final_row[key]:.6f}")
    print(f"모델 저장/재로드 예측 동일: {reload_equal}")
    for target in targets:
        print(f"생성 파일: {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

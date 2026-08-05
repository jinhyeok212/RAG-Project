"""기존 baseline 검색 결과로 실패 원인 후보와 세부 통계를 생성합니다."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval_utils import clean_text

BASELINE_DIR = PROJECT_ROOT / "results" / "retrieval"
OUTPUT_DIR = PROJECT_ROOT / "results" / "failure_analysis"


def require_columns(frame: pd.DataFrame, names: Iterable[str], label: str) -> None:
    """실제 CSV에 필요한 컬럼이 있는지 확인합니다."""
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"{label}에 필요한 컬럼이 없습니다: {missing}")


def has_product_candidate(text: str) -> bool:
    """개체 라벨이 없으므로 상품 표현으로 보이는 문자열을 보수적으로 탐지합니다."""
    terms = (
        "우유", "라면", "과자", "맥주", "소주", "와인", "김치", "쌀", "고기", "계란",
        "달걀", "커피", "음료", "세제", "샴푸", "치약", "휴지", "과일", "사과", "배",
        "딸기", "수박", "포도", "두부", "빵", "케이크", "아이스크림", "상품", "제품",
    )
    return any(term in text for term in terms)


def entity_flags(text: str) -> dict[str, bool]:
    """질문 문자열에서 가격·수량·시간 등의 표현 후보를 규칙으로 찾습니다."""
    return {
        "product_name_candidate": has_product_candidate(text),
        "price_info_candidate": bool(re.search(r"\d[\d,]*(?:원|만원)|가격|얼마|비용|요금", text)),
        "quantity_info_candidate": bool(re.search(r"\d+\s*(?:개|병|봉|박스|팩|세트|kg|g|리터|L|인분)|몇\s*(?:개|병|봉|박스)", text, re.I)),
        "time_or_date_candidate": bool(re.search(r"\d+\s*(?:시|분|일|월|년)|오늘|내일|어제|오전|오후|주말|평일|언제|기간", text)),
        "demonstrative_candidate": any(token in text for token in ("이거", "그거", "저거", "이것", "그것", "저것", "이 제품", "그 제품", "그걸", "이걸")),
    }


def similar_intent(expected: str, retrieved: str) -> bool:
    """인텐트 이름의 도메인 또는 행위 토큰이 겹치는지 확인합니다."""
    expected_tokens = {x for x in expected.split("_") if x}
    retrieved_tokens = {x for x in retrieved.split("_") if x}
    return bool(expected_tokens & retrieved_tokens)


def classify(row: pd.Series, rare_threshold: float, close_threshold: float, conflict_questions: set[str]) -> tuple[str, str]:
    """확정 원인이 아닌 검토 후보를 우선순위 규칙으로 분류합니다."""
    query = row["query"]
    expected = row["expected_intent"]
    top1 = row["top1_intent"]
    if row["query_length"] <= 10 or re.fullmatch(r"[아어네예음그래요\s?!.]+", query):
        return "짧거나 모호한 질문", "질문이 10자 이하이거나 독립적인 정보가 적은 표현 후보"
    if row["demonstrative_candidate"]:
        return "이전 대화 맥락 부족", "대명사·지시어가 있어 이전 발화 없이 대상 파악이 어려울 가능성"
    if query in conflict_questions:
        return "동일하거나 유사한 질문의 답변 충돌", "Training에서 같은 질문에 서로 다른 답변이 존재"
    if row["training_intent_count"] <= rare_threshold:
        return "Training 표본 부족", "기대 인텐트의 Training 문서 수가 하위 25% 구간"
    if similar_intent(expected, top1):
        expected_action = expected.split("_")[-1]
        top1_action = top1.split("_")[-1]
        if expected_action != top1_action:
            return "인텐트 라벨 세분화 문제 가능성", "도메인은 겹치지만 질문·요청·확인 등 세부 라벨이 다름"
        return "유사 인텐트 혼동", "기대 인텐트와 Top1 인텐트 이름에 공통 의미 토큰이 있음"
    if not any((row["product_name_candidate"], row["price_info_candidate"], row["quantity_info_candidate"], row["time_or_date_candidate"])):
        return "상품명 또는 개체 정보 부족", "질문 문자열에서 상품·가격·수량·시간 정보 후보를 찾지 못함"
    if row["top1_distance"] <= close_threshold:
        return "질문 표현 차이", "벡터 거리는 상대적으로 가깝지만 기대 인텐트와 Top1 인텐트가 다름"
    return "기타 검토 필요", "현재 규칙만으로 뚜렷한 후보를 지정하기 어려움"


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    """외부 패키지 없이 작은 Markdown 표를 만듭니다."""
    selected = frame[columns] if limit is None else frame[columns].head(limit)
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for values in selected.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ")[:180] for value in values) + " |")
    return "\n".join([header, separator, *rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="새 실패 분석 산출물을 의도적으로 교체")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        OUTPUT_DIR / "super_failure_analysis_all.csv",
        OUTPUT_DIR / "super_failure_type_summary.csv",
        OUTPUT_DIR / "super_intent_confusion_pairs.csv",
        OUTPUT_DIR / "super_query_length_metrics.csv",
        OUTPUT_DIR / "super_manual_review_sample_30.csv",
        OUTPUT_DIR / "super_training_count_performance.csv",
        PROJECT_ROOT / "docs" / "retrieval_failure_analysis.md",
    ]
    existing = [str(path) for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError("기존 결과가 있어 중단합니다. 확인 후 --force를 사용하세요:\n" + "\n".join(existing))

    print("[1/6] 기존 baseline CSV의 실제 컬럼과 행 수 확인")
    train = pd.read_csv(PROJECT_ROOT / "data/experiment/super_train_sample_5000.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    validation = pd.read_csv(PROJECT_ROOT / "data/experiment/super_validation_sample_500.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    details = pd.read_csv(BASELINE_DIR / "super_validation_retrieval_results.csv", encoding="utf-8-sig")
    summaries = pd.read_csv(BASELINE_DIR / "super_validation_query_summary.csv", encoding="utf-8-sig")
    intent_metrics = pd.read_csv(BASELINE_DIR / "super_intent_metrics.csv", encoding="utf-8-sig")
    baseline_metrics = pd.read_csv(BASELINE_DIR / "super_retrieval_metrics.csv", encoding="utf-8-sig")
    require_columns(train, ["intent", "question", "answer"], "Training")
    require_columns(validation, ["query_id", "question", "expected_intent"], "Validation")
    require_columns(details, ["query_id", "rank", "retrieved_question", "retrieved_intent", "distance"], "상세 검색 결과")
    require_columns(summaries, ["query_id", "hit_at_1", "hit_at_3", "hit_at_5", "reciprocal_rank"], "질문 요약")

    print("[2/6] Hit@5 실패 질문 162개 특징 및 상위 5개 검색 결과 결합")
    training_counts = train.assign(intent=train["intent"].map(clean_text)).groupby("intent").size().to_dict()
    conflicts = set(
        train.groupby("question")["answer"].nunique().loc[lambda x: x > 1].index.astype(str)
    )
    top1 = details.loc[details["rank"] == 1, ["query_id", "retrieved_question", "retrieved_intent", "distance"]].rename(columns={
        "retrieved_question": "top1_question", "retrieved_intent": "top1_intent", "distance": "top1_distance"
    })
    top5_text = (
        details.loc[details["rank"] <= 5]
        .assign(item=lambda x: x.apply(lambda r: f"{int(r['rank'])}: {r['retrieved_question']} [{r['retrieved_intent']}] distance={r['distance']:.6f}", axis=1))
        .groupby("query_id")["item"].agg(lambda values: " || ".join(values)).rename("top5_results")
    )
    base = validation.rename(columns={"question": "query"}).merge(summaries.drop(columns=["query", "expected_intent"], errors="ignore"), on="query_id", how="inner")
    failures = base.loc[base["hit_at_5"] == 0].merge(top1, on="query_id", how="left").merge(top5_text, on="query_id", how="left")
    failures["query_length"] = failures["query"].str.len()
    for name in ("expected_intent", "top1_intent", "query", "top1_question"):
        failures[name] = failures[name].map(clean_text)
    flags = pd.DataFrame([entity_flags(text) for text in failures["query"]], index=failures.index)
    failures = pd.concat([failures, flags], axis=1)
    failures["training_intent_count"] = failures["expected_intent"].map(training_counts).fillna(0).astype(int)
    failures["intent_semantically_similar_candidate"] = failures.apply(lambda r: similar_intent(r["expected_intent"], r["top1_intent"]), axis=1)
    rare_threshold = float(pd.Series(list(training_counts.values())).quantile(0.25))
    close_threshold = float(failures["top1_distance"].quantile(0.25))
    classified = failures.apply(lambda r: classify(r, rare_threshold, close_threshold, conflicts), axis=1)
    failures["failure_type_candidate"] = [x[0] for x in classified]
    failures["failure_reason_candidate"] = [x[1] for x in classified]
    failures["manual_review_note"] = ""
    all_columns = [
        "query_id", "query", "expected_intent", "query_length", "top1_question", "top1_intent", "top1_distance",
        "top5_results", "product_name_candidate", "price_info_candidate", "quantity_info_candidate",
        "time_or_date_candidate", "demonstrative_candidate", "training_intent_count",
        "intent_semantically_similar_candidate", "first_matching_rank", "failure_type_candidate",
        "failure_reason_candidate", "manual_review_note",
    ]
    failures[all_columns].to_csv(targets[0], index=False, encoding="utf-8-sig")

    print("[3/6] 실패 유형, 인텐트 혼동, 질문 길이 통계 계산")
    summary_rows = []
    for failure_type, group in failures.groupby("failure_type_candidate"):
        summary_rows.append({
            "failure_type_candidate": failure_type,
            "question_count": len(group),
            "failure_ratio_percent": 100 * len(group) / len(failures),
            "average_query_length": group["query_length"].mean(),
            "average_top1_distance": group["top1_distance"].mean(),
            "major_expected_intents": " | ".join(group["expected_intent"].value_counts().head(3).index),
            "major_top1_intents": " | ".join(group["top1_intent"].value_counts().head(3).index),
            "representative_queries": " | ".join(group["query"].head(3)),
        })
    type_summary = pd.DataFrame(summary_rows).sort_values("question_count", ascending=False)
    type_summary.to_csv(targets[1], index=False, encoding="utf-8-sig")

    confusion = failures.groupby(["expected_intent", "top1_intent"], as_index=False).size().rename(columns={"size": "confusion_count"})
    confusion["ratio_of_failures_percent"] = 100 * confusion["confusion_count"] / len(failures)
    confusion["intent_name_similarity_candidate"] = confusion.apply(lambda r: similar_intent(r["expected_intent"], r["top1_intent"]), axis=1)
    confusion = confusion.sort_values(["confusion_count", "expected_intent", "top1_intent"], ascending=[False, True, True])
    confusion["confusion_rank"] = np.arange(1, len(confusion) + 1)
    confusion.to_csv(targets[2], index=False, encoding="utf-8-sig")

    top1_all = details.loc[details["rank"] == 1, ["query_id", "distance"]].rename(columns={"distance": "top1_distance"})
    length_data = base.merge(top1_all, on="query_id", how="left")
    length_data["query_length"] = length_data["query"].str.len()
    length_data["length_bucket"] = pd.cut(length_data["query_length"], bins=[4, 10, 20, 40, np.inf], labels=["5~10자", "11~20자", "21~40자", "41자 이상"])
    length_metrics = length_data.groupby("length_bucket", observed=False).agg(
        question_count=("query_id", "count"), intent_hit_at_1=("hit_at_1", "mean"),
        intent_hit_at_3=("hit_at_3", "mean"), intent_hit_at_5=("hit_at_5", "mean"),
        intent_mrr=("reciprocal_rank", "mean"), average_top1_distance=("top1_distance", "mean"),
    ).reset_index()
    length_metrics.to_csv(targets[3], index=False, encoding="utf-8-sig")

    print("[4/6] Training 표본 수와 인텐트 성능 관계 계산")
    relation = intent_metrics.copy()
    relation["training_intent_count"] = relation["expected_intent"].map(training_counts).fillna(0).astype(int)
    pearson = relation["training_intent_count"].corr(relation["intent_hit_at_5"], method="pearson")
    spearman = relation["training_intent_count"].corr(relation["intent_hit_at_5"], method="spearman")
    relation.to_csv(targets[5], index=False, encoding="utf-8-sig")

    print("[5/6] 서로 다른 특징을 고르게 포함한 수동 검토 표본 30개 선정")
    review = []
    used: set[str] = set()
    selectors = [
        failures["query_length"] <= 10,
        failures["demonstrative_candidate"],
        failures["intent_semantically_similar_candidate"],
        failures["product_name_candidate"],
        failures["price_info_candidate"] | failures["quantity_info_candidate"],
        failures["training_intent_count"] <= rare_threshold,
        failures["top1_distance"] <= failures["top1_distance"].quantile(0.25),
        failures["top1_distance"] >= failures["top1_distance"].quantile(0.75),
    ]
    while len(review) < 30:
        progressed = False
        for selector in selectors:
            candidates = failures.loc[selector & ~failures["query_id"].isin(used)].sort_values("top1_distance")
            if not candidates.empty and len(review) < 30:
                selected = candidates.iloc[len(review) % len(candidates)]
                review.append(selected); used.add(selected["query_id"]); progressed = True
        if not progressed:
            break
    if len(review) < 30:
        for _, selected in failures.loc[~failures["query_id"].isin(used)].head(30 - len(review)).iterrows():
            review.append(selected)
    review_frame = pd.DataFrame(review)[all_columns]
    review_frame["manual_review_note"] = ""
    review_frame.to_csv(targets[4], index=False, encoding="utf-8-sig")

    print("[6/6] 실제 통계를 문서로 작성")
    baseline = baseline_metrics.iloc[0]
    top1_over = details.loc[details["rank"] == 1, "retrieved_intent"].value_counts().head(10).rename_axis("retrieved_intent").reset_index(name="top1_count")
    expected_spread = failures.groupby("expected_intent")["top1_intent"].nunique().sort_values(ascending=False).head(10).rename("different_top1_intent_count").reset_index()
    document = f"""# Retrieval 실패 원인 분석

## baseline과 실패 후보의 의미

- Intent Hit@1: {baseline['intent_hit_at_1']:.6f}
- Intent Hit@3: {baseline['intent_hit_at_3']:.6f}
- Intent Hit@5: {baseline['intent_hit_at_5']:.6f}
- Intent MRR: {baseline['intent_mrr']:.6f}
- Hit@5 실패 질문: {len(failures)}개

실패 후보는 상위 5개 안에 기대 인텐트가 없었던 질문이다. 실제 답변의 정오를 확정한 것이 아니며, 인텐트 대리 평가 자체의 한계가 있다. 자동 실패 유형 역시 문자열 규칙으로 만든 **검토 후보**이지 정답 원인이 아니다.

## 실패 유형별 통계

{markdown_table(type_summary.round(4), list(type_summary.columns))}

## 주요 인텐트 혼동 상위 20개

{markdown_table(confusion, ['confusion_rank', 'expected_intent', 'top1_intent', 'confusion_count', 'intent_name_similarity_candidate'], 20)}

인텐트 이름의 토큰이 겹치는 조합은 의미가 비슷할 가능성이 있지만 이름만으로 의미 동일성을 확정할 수 없다. 특정 기대 인텐트가 여러 인텐트로 흩어지는 정도와 Top1에 과도하게 등장하는 인텐트는 아래 표로 확인했다.

### 여러 다른 인텐트로 오검색된 기대 인텐트

{markdown_table(expected_spread, list(expected_spread.columns), 10)}

### 전체 Validation Top1에 자주 등장한 인텐트

{markdown_table(top1_over, list(top1_over.columns), 10)}

## Training 표본 수와 성능 관계

- Training 인텐트 문서 수와 Hit@5의 Pearson 상관계수: {pearson:.6f}
- Spearman 순위 상관계수: {spearman:.6f}
- 계산 인텐트 수: {len(relation)}개

상관계수는 인과관계를 뜻하지 않는다. Training 표본이 인텐트별 최대치로 제한되어 범위가 좁고, Validation 질문 수가 매우 적은 인텐트는 Hit 비율 변동이 크므로 통계 해석에 주의해야 한다.

## 질문 길이별 실제 성능

{markdown_table(length_metrics.round(6), list(length_metrics.columns))}

질문이 짧을수록 무조건 낮다고 가정하지 않고 위 실제 지표로 비교했다. 평균 거리는 Chroma가 반환한 원래 거리이며 낮을수록 가깝다.

## 대표 실패 사례 30개

`results/failure_analysis/super_manual_review_sample_30.csv`에 짧은 질문, 지시어, 유사 인텐트, 상품·가격·수량 후보, 희소 인텐트, 가까운 거리와 먼 거리 사례를 가능한 범위에서 고르게 저장했다. `manual_review_note`는 사람이 작성하도록 비워 두었다.

## 개체 탐지와 자동 분류의 한계

현재 5,000개 및 500개 표본에는 원본의 `상품명`, `가격`, `수량`, `크기`, `시간`, `날짜` 컬럼이 없다. 따라서 이 분석의 `*_candidate` 값은 질문 문자열 정규식과 제한된 단어 목록으로 탐지한 후보이며 실제 개체 라벨이 아니다. 상위 5개 검색 결과, 원 상담 문맥, 라벨 정의와 답변 충돌 여부를 사람이 직접 확인해야 한다.

## 실행 방법

```powershell
C:\\Python312\\python.exe scripts\\analyze_retrieval_failures.py
```

기존 새 분석 결과를 확인하고 재생성할 때만 `--force`를 추가한다. 기존 baseline 결과 파일은 수정하지 않는다.
"""
    (PROJECT_ROOT / "docs/retrieval_failure_analysis.md").write_text(document, encoding="utf-8-sig")
    print(f"완료: 실패 {len(failures)}개, 수동 검토 {len(review_frame)}개")
    for target in targets:
        print(f"생성 파일: {target.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

# 예측 인텐트 제한 Retrieval 실험

## 왜 인텐트 분류기가 필요한가?

Training에는 정답 `intent`가 있지만 실제 사용자의 새 질문에는 라벨이 없다. 정답 인텐트를 검색 필터에 직접 넣으면 평가 정답을 미리 사용한 데이터 누수다. 이 실험은 Training 질문과 기존 인텐트 이름만 학습한 분류기가 새 질문의 인텐트를 예측하고, 검색기에는 그 **예측값만** 전달한다. Validation의 `expected_intent`는 검색이 끝난 뒤 지표 계산에만 사용했다.

## 분류기

TF-IDF와 Logistic Regression을 sklearn Pipeline으로 구성했다. Training 내부 홀드아웃에서 char n-gram balanced/unbalanced와 word n-gram balanced를 비교하고 Macro F1이 가장 높은 설정을 선택한 뒤 전체 5,000개로 재학습했다. Validation은 최종 평가에만 사용했다. 샘플 1개짜리 인텐트도 삭제하지 않았으며 내부 홀드아웃에는 넣지 않고 학습에 남겼다.

| 지표 | 실제 값 |
|---|---:|
| 선택 설정 | char_balanced |
| Accuracy / Top-1 | 0.502000 |
| Macro F1 | 0.397292 |
| Weighted F1 | 0.500151 |
| Top-3 Accuracy | 0.728000 |
| Top-5 Accuracy | 0.812000 |
| 저장 모델 재로드 동일 | True |

## 검색 전략

- `full`: 기존 5,000개 전체 검색
- `top1_strict`: 예측 Top-1 인텐트만 검색
- `top1_fallback`: Top-1 제한 결과가 5개 미만일 때 전체 검색으로 보충
- `top3_strict`: 예측 Top-3 인텐트 중에서 검색
- `top3_fallback`: Top-3 제한 결과가 5개 미만일 때 전체 검색으로 보충

Strict는 분류 오류에 민감하다. Fallback은 희소 인텐트에서 문서 수 부족을 보완하지만 잘못된 전체 검색 문서가 다시 들어올 수 있다. 모든 결과는 document_id 중복 제거 후 Chroma 원래 distance 오름차순으로 정렬했다. distance는 유사도가 아니며 낮을수록 가깝다.

## 실제 비교

| strategy | intent_hit_at_1 | intent_hit_at_3 | intent_hit_at_5 | intent_mrr | hit_at_5_failure_count | average_search_time_ms | p50_search_time_ms | p95_search_time_ms | average_returned_documents |
|---|---|---|---|---|---|---|---|---|---|
| full | 0.452000 | 0.604000 | 0.676000 | 0.551851 | 162 | 8.644236 | 7.658500 | 12.344125 | 5.000000 |
| top1_strict | 0.502000 | 0.502000 | 0.502000 | 0.502000 | 249 | 15.829627 | 14.151550 | 23.316315 | 4.958000 |
| top1_fallback | 0.510000 | 0.514000 | 0.514000 | 0.511994 | 243 | 14.419440 | 13.772850 | 20.104430 | 5.000000 |
| top3_strict | 0.552000 | 0.658000 | 0.708000 | 0.614777 | 146 | 17.918623 | 18.121150 | 23.366400 | 5.000000 |
| top3_fallback | 0.552000 | 0.658000 | 0.708000 | 0.614777 | 146 | 16.109497 | 15.258900 | 22.936980 | 5.000000 |

기존 참고 baseline은 Hit@1=0.452, Hit@3=0.604, Hit@5=0.676, MRR=0.551851이다. 위 `full`은 기존 컬렉션에서 동일 500개를 다시 검색한 실제 값이며 결과가 낮더라도 그대로 기록했다. MRR은 기존 실험과 맞춰 최대 20위에서 첫 인텐트 일치를 확인했고 상세 CSV에는 최종 Top-5만 저장했다.

실측 기준 가장 높은 Hit@5 전략은 **top3_fallback**이다. 다만 단일 수치만으로 운영 전략을 확정하지 말고 분류 오류 시 실패, 지연, 반환 문서 수와 fallback 동작을 함께 봐야 한다. 기본 RAG는 요청대로 여전히 `full`이며 자동 변경하지 않았다.

## 실패 사례 유형

| failure_case_type | case_count |
|---|---|
| Top1 예측 오답·제한검색 실패 | 249 |
| 낮은 확률 또는 근접 확률 | 163 |
| Full 성공·Top1 제한 실패 | 126 |
| Top1 실패·Top3 성공 | 105 |
| Full 실패·제한검색 성공 | 67 |

낮은 확률은 임의의 고정 임계값이 아니라 이번 500개 분포의 하위 25%를 사용했다. Top-1 확률 기준은 0.011913, Top1-Top2 확률 차이 기준은 0.001211이다. 유형은 서로 겹칠 수 있어 합계가 500을 넘을 수 있으며 자동 원인 확정이 아니다.

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
C:\Python312\python.exe scripts\train_intent_classifier.py
C:\Python312\python.exe scripts\evaluate_intent_filtered_retrieval.py
```

의도적으로 새 결과를 교체할 때만 두 명령에 `--force`를 추가한다. 기본 RAG/LLM 코드는 변경하지 않았다.

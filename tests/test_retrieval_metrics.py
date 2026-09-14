"""Document와 Chunk가 공통으로 사용하는 Hit@k, RR, MRR 계산 테스트."""

import unittest

from src.evaluation.retrieval_metrics import (
    calculate_hit_at_k,
    calculate_mean_hit_at_k,
    calculate_mrr,
    calculate_reciprocal_rank,
    find_first_relevant_rank,
)


class RetrievalMetricsTest(unittest.TestCase):
    """ID 종류와 관계없이 공통 검색 평가지표 계산이 정확한지 확인한다."""

    # 정답 ID가 1위에 있을 때의 평가지표를 확인한다.
    def test_gold_id_at_rank_1(self) -> None:
        first_rank = find_first_relevant_rank(
            ["정책_A", "정책_B", "정책_C"],
            {"정책_A"},
        )

        self.assertEqual(first_rank, 1)
        self.assertEqual(calculate_hit_at_k(first_rank, 1), 1)
        self.assertEqual(calculate_hit_at_k(first_rank, 3), 1)
        self.assertEqual(calculate_hit_at_k(first_rank, 5), 1)
        self.assertEqual(calculate_reciprocal_rank(first_rank), 1.0)

    # 정답 ID가 4위에 있을 때의 평가지표를 확인한다.
    def test_gold_id_at_rank_4(self) -> None:
        first_rank = find_first_relevant_rank(
            ["정책_A", "정책_B", "정책_C", "정책_D", "정책_E"],
            {"정책_D"},
        )

        self.assertEqual(first_rank, 4)
        self.assertEqual(calculate_hit_at_k(first_rank, 1), 0)
        self.assertEqual(calculate_hit_at_k(first_rank, 3), 0)
        self.assertEqual(calculate_hit_at_k(first_rank, 5), 1)
        self.assertEqual(calculate_reciprocal_rank(first_rank), 0.25)

    # 정답 ID가 Top-5에 없을 때 모든 Hit@k와 RR이 0인지 확인한다.
    def test_gold_id_not_retrieved(self) -> None:
        first_rank = find_first_relevant_rank(
            ["정책_A", "정책_B", "정책_C", "정책_D", "정책_E"],
            {"정책_F"},
        )

        self.assertIsNone(first_rank)
        self.assertEqual(calculate_hit_at_k(first_rank, 1), 0)
        self.assertEqual(calculate_hit_at_k(first_rank, 3), 0)
        self.assertEqual(calculate_hit_at_k(first_rank, 5), 0)
        self.assertEqual(calculate_reciprocal_rank(first_rank), 0.0)

    # Gold ID가 여러 개면 가장 먼저 검색된 Gold의 순위를 사용하는지 확인한다.
    def test_multiple_gold_ids_use_first_match(self) -> None:
        first_rank = find_first_relevant_rank(
            ["정책_A", "정책_B", "정책_C", "정책_D"],
            {"정책_B", "정책_D"},
        )

        self.assertEqual(first_rank, 2)
        self.assertEqual(calculate_reciprocal_rank(first_rank), 0.5)

    # 질문별 Hit@k의 평균이 올바르게 계산되는지 확인한다.
    def test_calculate_mean_hit_at_k(self) -> None:
        self.assertAlmostEqual(
            calculate_mean_hit_at_k([1, 1, 0, 1]),
            0.75,
        )

    # 질문별 RR의 평균인 MRR이 올바르게 계산되는지 확인한다.
    def test_calculate_mrr(self) -> None:
        self.assertAlmostEqual(
            calculate_mrr([1.0, 0.5, 0.25, 0.0]),
            0.4375,
        )

    # 계산할 값이 없을 때 잘못된 평균을 만들지 않고 중단하는지 확인한다.
    def test_empty_metric_values_raise_error(self) -> None:
        with self.assertRaises(ValueError):
            calculate_mean_hit_at_k([])

        with self.assertRaises(ValueError):
            calculate_mrr([])


if __name__ == "__main__":
    unittest.main()

"""실제 Baseline 입력 파일을 사용하는 Document 평가 통합 테스트."""

import unittest
import json
import tempfile
from pathlib import Path

from src.evaluation.evaluator import (
    create_evaluation_manifest,
    evaluate_document_queries,
    load_and_validate_inputs,
    run_document_evaluation,
    save_document_evaluation,
    summarize_document_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_RESULTS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "exp_baseline_kure_v1"
    / "retrieval_results.jsonl"
)
QRELS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ontong_youth_eval_questions_v2_final_47"
    / "qrels.jsonl"
)


class DocumentEvaluatorIntegrationTest(unittest.TestCase):
    """KURE-v1 Baseline의 실제 Document 평가 흐름을 확인한다."""

    @classmethod
    def setUpClass(cls) -> None:
        # 실제 검색 결과와 qrels를 읽고 query_id 연결을 확인한다.
        retrieval_results, document_qrels = load_and_validate_inputs(
            RETRIEVAL_RESULTS_PATH,
            QRELS_PATH,
        )

        # 47개 질문의 개별 평가 결과와 전체 요약을 만든다.
        cls.query_metrics = evaluate_document_queries(
            retrieval_results,
            document_qrels,
        )
        cls.summary = summarize_document_metrics(cls.query_metrics)
        cls.manifest = create_evaluation_manifest(
            retrieval_results,
            document_qrels,
        )

    # Retrieval 결과와 qrels의 47개 질문이 모두 평가되는지 확인한다.
    def test_all_47_queries_are_evaluated(self) -> None:
        self.assertEqual(len(self.query_metrics), 47)
        self.assertEqual(self.summary["question_count"], 47)

    # 실제 KURE-v1 Baseline의 전체 Document 평가지표를 확인한다.
    def test_baseline_document_summary(self) -> None:
        self.assertAlmostEqual(
            self.summary["document_hit_at_1"],
            42 / 47,
        )
        self.assertAlmostEqual(
            self.summary["document_hit_at_3"],
            46 / 47,
        )
        self.assertAlmostEqual(
            self.summary["document_hit_at_5"],
            46 / 47,
        )
        self.assertAlmostEqual(
            self.summary["document_mrr"],
            0.925531914893617,
        )

    # Top-5 성공 46개와 실패 1개가 정확히 집계되는지 확인한다.
    def test_top_5_success_and_failure_counts(self) -> None:
        self.assertEqual(self.summary["top_5_success_count"], 46)
        self.assertEqual(self.summary["top_5_failure_count"], 1)

    # 실제 Top-5 실패 질문이 평가 결과에 올바르게 표시되는지 확인한다.
    def test_failed_query_metrics(self) -> None:
        failed_queries = [
            result
            for result in self.query_metrics
            if result["hit_at_5"] == 0
        ]

        self.assertEqual(len(failed_queries), 1)
        self.assertEqual(failed_queries[0]["query_id"], "oy_eval_v2r_q0090")
        self.assertIsNone(failed_queries[0]["first_relevant_rank"])
        self.assertEqual(failed_queries[0]["reciprocal_rank"], 0.0)

    # 저장한 질문별 결과와 전체 요약을 다시 읽을 수 있는지 확인한다.
    def test_save_document_evaluation(self) -> None:
        # 테스트 도구의 사용자 Temp 접근 제한을 피하려고 프로젝트 내부에 만든다.
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary_directory:
            metrics_path, summary_path, manifest_path = save_document_evaluation(
                self.query_metrics,
                self.summary,
                self.manifest,
                temporary_directory,
            )

            with metrics_path.open("r", encoding="utf-8") as metrics_file:
                saved_metrics = [
                    json.loads(line)
                    for line in metrics_file
                    if line.strip()
                ]

            with summary_path.open("r", encoding="utf-8") as summary_file:
                saved_summary = json.load(summary_file)

            with manifest_path.open("r", encoding="utf-8") as manifest_file:
                saved_manifest = json.load(manifest_file)

            self.assertEqual(len(saved_metrics), 47)
            self.assertEqual(saved_metrics[0], self.query_metrics[0])
            self.assertEqual(saved_summary, self.summary)
            self.assertEqual(saved_manifest, self.manifest)

    # Manifest에 Baseline 평가 조건과 데이터 건수가 기록되는지 확인한다.
    def test_evaluation_manifest(self) -> None:
        self.assertEqual(self.manifest["experiment_id"], "exp_baseline_kure_v1")
        self.assertEqual(self.manifest["evaluation_scope"], "document")
        self.assertEqual(self.manifest["embedding_model"], "nlpai-lab/KURE-v1")
        self.assertEqual(self.manifest["evaluation_max_k"], 5)
        self.assertEqual(self.manifest["question_count"], 47)
        self.assertEqual(self.manifest["qrels_count"], 50)

    # Document 평가 전체 과정이 한 번에 실행되고 세 파일이 생성되는지 확인한다.
    def test_run_document_evaluation(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary_directory:
            query_metrics, summary, manifest = run_document_evaluation(
                RETRIEVAL_RESULTS_PATH,
                QRELS_PATH,
                temporary_directory,
            )

            output_path = Path(temporary_directory)
            self.assertEqual(len(query_metrics), 47)
            self.assertEqual(summary["top_5_failure_count"], 1)
            self.assertEqual(manifest["experiment_id"], "exp_baseline_kure_v1")
            self.assertTrue((output_path / "document_metrics.jsonl").is_file())
            self.assertTrue((output_path / "document_summary.json").is_file())
            self.assertTrue(
                (output_path / "document_evaluation_manifest.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()

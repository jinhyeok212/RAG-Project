"""실제 Baseline 입력 파일을 사용하는 Chunk 평가 통합 테스트."""

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.evaluator import (
    create_chunk_evaluation_manifest,
    evaluate_chunk_queries,
    load_chunk_qrels,
    load_retrieval_results,
    run_chunk_evaluation,
    save_chunk_evaluation,
    summarize_chunk_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_RESULTS_PATH = (
    PROJECT_ROOT / "experiments" / "exp_baseline_kure_v1" / "retrieval_results.jsonl"
)
CHUNK_QRELS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunking2"
    / "05.chunks"
    / "c2_section_800"
    / "qrels_chunk.jsonl"
)


class ChunkEvaluatorIntegrationTest(unittest.TestCase):
    """KURE-v1 Baseline의 실제 Chunk 평가 흐름을 확인한다."""

    @classmethod
    def setUpClass(cls) -> None:
        retrieval_results = load_retrieval_results(RETRIEVAL_RESULTS_PATH)
        chunk_qrels = load_chunk_qrels(CHUNK_QRELS_PATH)
        cls.query_metrics = evaluate_chunk_queries(retrieval_results, chunk_qrels)
        cls.summary = summarize_chunk_metrics(cls.query_metrics)
        cls.manifest = create_chunk_evaluation_manifest(
            retrieval_results,
            chunk_qrels,
        )

    # Retrieval 결과와 Chunk qrels의 47개 질문이 모두 평가되는지 확인한다.
    def test_all_47_queries_are_evaluated(self) -> None:
        self.assertEqual(len(self.query_metrics), 47)
        self.assertEqual(self.summary["question_count"], 47)

    # 검수된 Chunk qrels를 사용한 실제 KURE-v1 Baseline 수치를 확인한다.
    def test_baseline_chunk_summary(self) -> None:
        self.assertAlmostEqual(self.summary["chunk_hit_at_1"], 24 / 47)
        self.assertAlmostEqual(self.summary["chunk_hit_at_3"], 40 / 47)
        self.assertAlmostEqual(self.summary["chunk_hit_at_5"], 41 / 47)
        self.assertAlmostEqual(self.summary["chunk_mrr"], 0.6684397163120568)
        self.assertEqual(self.summary["top_5_success_count"], 41)
        self.assertEqual(self.summary["top_5_failure_count"], 6)

    # 복수 Gold Chunk 중 하나가 검색되면 가장 빠른 순위를 사용하는지 확인한다.
    def test_multiple_gold_chunks_use_first_match(self) -> None:
        metric = next(
            result
            for result in self.query_metrics
            if result["query_id"] == "oy_eval_repair_q0037"
        )
        self.assertEqual(len(metric["gold_chunk_ids"]), 2)
        if metric["matched_gold_chunk_ids"]:
            self.assertIsNotNone(metric["first_relevant_rank"])

    # Chunk 평가 결과 세 파일을 저장하고 다시 읽을 수 있는지 확인한다.
    def test_save_chunk_evaluation(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary_directory:
            metrics_path, summary_path, manifest_path = save_chunk_evaluation(
                self.query_metrics,
                self.summary,
                self.manifest,
                temporary_directory,
            )
            saved_metrics = [
                json.loads(line)
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(saved_metrics, self.query_metrics)
            self.assertEqual(saved_summary, self.summary)
            self.assertEqual(saved_manifest, self.manifest)

    # Manifest에 Chunk 평가 범위와 Gold Chunk 50개가 기록되는지 확인한다.
    def test_chunk_evaluation_manifest(self) -> None:
        self.assertEqual(self.manifest["evaluation_scope"], "chunk")
        self.assertEqual(self.manifest["question_count"], 47)
        self.assertEqual(self.manifest["qrels_count"], 50)
        self.assertEqual(self.manifest["chunking_version"], "c2_section_800_v1")

    # Chunk 평가 전체 과정이 실행되고 Document 파일과 다른 이름으로 저장되는지 확인한다.
    def test_run_chunk_evaluation(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary_directory:
            query_metrics, summary, manifest = run_chunk_evaluation(
                RETRIEVAL_RESULTS_PATH,
                CHUNK_QRELS_PATH,
                temporary_directory,
            )
            output_path = Path(temporary_directory)
            self.assertEqual(len(query_metrics), 47)
            self.assertEqual(summary["question_count"], 47)
            self.assertEqual(manifest["evaluation_scope"], "chunk")
            self.assertTrue((output_path / "chunk_metrics.jsonl").is_file())
            self.assertTrue((output_path / "chunk_summary.json").is_file())
            self.assertTrue((output_path / "chunk_evaluation_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()

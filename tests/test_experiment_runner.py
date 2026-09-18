"""Tests for Experiment Config, Contracts, and Runner."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiments.config_validator import (
    ConfigValidationError,
    DEFAULT_CONFIG_PATH,
    DEFAULT_SCHEMA_PATH,
    PROJECT_ROOT,
    validate_experiment_config,
    validate_semantics,
)
from src.experiments.contracts import (
    ContractError,
    validate_stage_result,
)
from src.experiments.runner import (
    prepare_output_directory,
    run_experiment,
)


class ExperimentConfigTest(unittest.TestCase):
    """공통 YAML과 추가 설정 규칙을 검사한다."""

    def test_shared_config_matches_schema(self) -> None:
        """공통 YAML이 Experiment Schema를 통과해야 한다."""

        config = validate_experiment_config(
            config_path=DEFAULT_CONFIG_PATH,
            schema_path=DEFAULT_SCHEMA_PATH,
            check_files=False,
            project_root=PROJECT_ROOT,
        )

        self.assertEqual(
            config["schema_version"],
            "1.0",
        )
        self.assertEqual(
            config["experiment_id"],
            "exp_full_kure_v1",
        )

    def test_overlap_must_be_smaller_than_size(
        self,
    ) -> None:
        """Overlap이 Chunk Size 이상이면 실패해야 한다."""

        config = validate_experiment_config(
            config_path=DEFAULT_CONFIG_PATH,
            schema_path=DEFAULT_SCHEMA_PATH,
            check_files=False,
            project_root=PROJECT_ROOT,
        )

        invalid_config = copy.deepcopy(config)

        invalid_config["chunking"]["overlap"] = (
            invalid_config["chunking"]["size"]
        )

        with self.assertRaises(
            ConfigValidationError
        ):
            validate_semantics(
                config=invalid_config,
                project_root=PROJECT_ROOT,
            )


class ExperimentContractTest(unittest.TestCase):
    """팀 모듈의 반환값 계약을 검사한다."""

    def test_dataset_contract_accepts_valid_result(
        self,
    ) -> None:
        """필수 필드를 포함한 DatasetInfo는 통과해야 한다."""

        result = validate_stage_result(
            stage="validate_documents",
            value={
                "document_path": "documents.jsonl",
                "manifest_path": (
                    "dataset_manifest.json"
                ),
                "document_count": 10,
                "sha256": "abc123",
            },
        )

        self.assertEqual(
            result["document_count"],
            10,
        )

    def test_dataset_contract_rejects_missing_fields(
        self,
    ) -> None:
        """필수 필드가 없으면 ContractError가 발생해야 한다."""

        with self.assertRaises(ContractError):
            validate_stage_result(
                stage="validate_documents",
                value={
                    "document_path": (
                        "documents.jsonl"
                    )
                },
            )


class ExperimentRunnerTest(unittest.TestCase):
    """Runner의 실행 순서와 결과 보호 기능을 검사한다."""

    def load_config(self) -> dict:
        """테스트에 사용할 공통 설정을 불러온다."""

        return validate_experiment_config(
            config_path=DEFAULT_CONFIG_PATH,
            schema_path=DEFAULT_SCHEMA_PATH,
            check_files=False,
            project_root=PROJECT_ROOT,
        )

    def test_existing_output_is_not_overwritten(
        self,
    ) -> None:
        """기존 결과 폴더가 있으면 덮어쓰지 않아야 한다."""

        config = self.load_config()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)

            output_directory = (
                temp_root
                / "experiments"
                / config["experiment_id"]
            )

            output_directory.mkdir(
                parents=True,
            )

            existing_result = (
                output_directory
                / "metrics_summary.json"
            )

            existing_result.write_text(
                "{}",
                encoding="utf-8",
            )

            with self.assertRaises(
                FileExistsError
            ):
                prepare_output_directory(
                    config=config,
                    project_root=temp_root,
                )

    def test_runner_calls_stages_in_order(
        self,
    ) -> None:
        """Runner가 합의한 순서대로 모듈을 호출해야 한다."""

        config = self.load_config()
        config = copy.deepcopy(config)

        calls: list[str] = []

        def validate_documents(
            received_config: dict,
        ) -> dict:
            calls.append(
                "validate_documents"
            )

            self.assertEqual(
                received_config["experiment_id"],
                "exp_full_kure_v1",
            )

            return {
                "document_path": (
                    "data/processed/test/"
                    "documents.jsonl"
                ),
                "manifest_path": (
                    "data/processed/test/"
                    "dataset_manifest.json"
                ),
                "document_count": 1,
                "sha256": "test-dataset-sha256",
            }

        def build_chunks(
            received_config: dict,
            dataset_info: dict,
        ) -> dict:
            calls.append(
                "build_chunks"
            )

            self.assertEqual(
                dataset_info["document_count"],
                1,
            )

            return {
                "chunk_path": (
                    "data/processed/test/"
                    "chunks.jsonl"
                ),
                "manifest_path": (
                    "data/processed/test/"
                    "chunk_manifest.json"
                ),
                "chunk_count": 1,
                "chunk_qrels_path": (
                    "data/evaluation/test/"
                    "qrels_chunk.jsonl"
                ),
            }

        def resolve_index(
            received_config: dict,
            chunk_info: dict,
        ) -> dict:
            calls.append(
                "resolve_index"
            )

            self.assertEqual(
                chunk_info["chunk_count"],
                1,
            )

            return {
                "index_version": "index_test_v1",
                "db_path": "indexes/chroma/test",
                "collection_name": (
                    "test_collection"
                ),
                "reused": False,
                "manifest_path": (
                    "experiments/test/"
                    "index_manifest.json"
                ),
                "index_count": 1,
                "build_time_seconds": 0.1,
            }

        def run_batch_retrieval(
            received_config: dict,
            index_info: dict,
        ) -> list[dict]:
            calls.append(
                "run_batch_retrieval"
            )

            self.assertEqual(
                index_info["collection_name"],
                "test_collection",
            )

            return [
                {
                    "query_id": "q001",
                    "query": "테스트 질문",
                    "retrieval_latency_ms": 1.0,
                    "results": [
                        {
                            "rank": 1,
                            "chunk_id": "chunk_001",
                            "document_id": "document_001",
                            "distance": 0.1,
                            "similarity": 0.9,
                        }
                    ],
                }
            ]

        def evaluate_experiment(
            received_config: dict,
            retrieval_results: list[dict],
        ) -> dict:
            calls.append(
                "evaluate_experiment"
            )

            self.assertEqual(
                len(retrieval_results),
                1,
            )

            return {
                "metrics_summary": {
                    "document_hit_at_1": 1.0,
                    "document_mrr": 1.0,
                    "chunk_hit_at_1": 1.0,
                    "chunk_mrr": 1.0,
                },
                "metrics_by_query": [
                    {
                        "query_id": "q001",
                        "document_hit_at_1": 1,
                        "chunk_hit_at_1": 1,
                    }
                ],
                "failure_cases": [],
            }

        def write_experiment_artifacts(
            received_config: dict,
            run_context: dict,
        ) -> dict:
            calls.append(
                "write_experiment_artifacts"
            )

            self.assertEqual(
                run_context[
                    "run_metadata"
                ]["status"],
                "COMPLETED",
            )

            return {
                "output_dir": (
                    "experiments/"
                    "exp_full_kure_v1"
                ),
                "written_files": [
                    "metrics_summary.json",
                    "run.log",
                ],
            }

        fake_functions = [
            validate_documents,
            build_chunks,
            resolve_index,
            run_batch_retrieval,
            evaluate_experiment,
            write_experiment_artifacts,
        ]

        expected_order = [
            "validate_documents",
            "build_chunks",
            "resolve_index",
            "run_batch_retrieval",
            "evaluate_experiment",
            "write_experiment_artifacts",
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)

            with patch(
                "src.experiments.runner.load_callable",
                side_effect=fake_functions,
            ):
                context = run_experiment(
                    config=config,
                    project_root=temp_root,
                )

            self.assertEqual(
                calls,
                expected_order,
            )

            self.assertEqual(
                context[
                    "run_metadata"
                ]["status"],
                "COMPLETED",
            )

            log_path = (
                temp_root
                / "experiments"
                / config["experiment_id"]
                / "run.log"
            )

            self.assertTrue(
                log_path.is_file()
            )

            log_text = log_path.read_text(
                encoding="utf-8"
            )

            self.assertIn(
                "experiment_completed",
                log_text,
            )


if __name__ == "__main__":
    unittest.main()
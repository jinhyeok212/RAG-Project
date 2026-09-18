"""Run the complete RAG experiment pipeline from one YAML file."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.experiments.config_validator import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_SCHEMA_PATH,
    PROJECT_ROOT,
    resolve_project_path,
    validate_experiment_config,
)
from src.experiments.contracts import (
    validate_stage_result,
)


@dataclass(frozen=True)
class PipelineStage:
    """Runner에서 실행할 Pipeline 단계 정보."""

    name: str
    description: str


PIPELINE_STAGES = (
    PipelineStage(
        name="validate_documents",
        description="문서와 Dataset Manifest 검증",
    ),
    PipelineStage(
        name="build_chunks",
        description="청킹 및 Chunk qrels 검증",
    ),
    PipelineStage(
        name="resolve_index",
        description="Chroma 인덱스 생성 또는 재사용",
    ),
    PipelineStage(
        name="run_batch_retrieval",
        description="전체 평가 질문 Top-k 검색",
    ),
    PipelineStage(
        name="evaluate_experiment",
        description="Hit@k, MRR 및 실패 사례 계산",
    ),
    PipelineStage(
        name="write_experiment_artifacts",
        description="표준 실험 결과 파일 저장",
    ),
)


def utc_now() -> str:
    """현재 UTC 시각을 ISO 8601 형식으로 반환한다."""

    return datetime.now(timezone.utc).isoformat()


def load_callable(
    entrypoint: str,
) -> Callable[..., Any]:
    """module:function 형식의 Entrypoint를 불러온다."""

    if ":" not in entrypoint:
        raise ValueError(
            "Entrypoint는 module:function 형식이어야 합니다: "
            f"{entrypoint}"
        )

    module_name, function_name = entrypoint.split(
        ":",
        1,
    )

    module = importlib.import_module(module_name)

    function = getattr(
        module,
        function_name,
        None,
    )

    if not callable(function):
        raise TypeError(
            "Entrypoint에서 호출 가능한 함수를 "
            f"찾을 수 없습니다: {entrypoint}"
        )

    return function


def inspect_input_paths(
    config: dict[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    """YAML에 정의된 입력 파일 준비 상태를 확인한다."""

    configured_paths = {
        "dataset.document_path": (
            config["dataset"]["document_path"]
        ),
        "dataset.chunk_path": (
            config["dataset"]["chunk_path"]
        ),
        "dataset.manifest_path": (
            config["dataset"]["manifest_path"]
        ),
        "evaluation.questions_path": (
            config["evaluation"]["questions_path"]
        ),
        "evaluation.document_qrels_path": (
            config["evaluation"]["document_qrels_path"]
        ),
        "evaluation.chunk_qrels_path": (
            config["evaluation"]["chunk_qrels_path"]
        ),
        "evaluation.manifest_path": (
            config["evaluation"]["manifest_path"]
        ),
    }

    results = []

    for field_name, configured_path in (
        configured_paths.items()
    ):
        resolved_path = resolve_project_path(
            configured_path=configured_path,
            project_root=project_root,
        )

        results.append(
            {
                "field": field_name,
                "path": configured_path,
                "exists": resolved_path.is_file(),
            }
        )

    return results


def inspect_pipeline_modules(
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """각 Pipeline Entrypoint의 import 가능 여부를 확인한다."""

    entrypoints = config["pipeline"]["entrypoints"]
    results = []

    for order, stage in enumerate(
        PIPELINE_STAGES,
        start=1,
    ):
        entrypoint = entrypoints[stage.name]

        try:
            load_callable(entrypoint)
            import_status = "READY"
            detail = None
        except Exception as exc:
            import_status = "WAITING_FOR_MODULE"
            detail = (
                f"{type(exc).__name__}: {exc}"
            )

        results.append(
            {
                "order": order,
                "stage": stage.name,
                "description": stage.description,
                "entrypoint": entrypoint,
                "import_status": import_status,
                "detail": detail,
            }
        )

    return results


def build_dry_run_report(
    config: dict[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """파일과 모듈 준비 상태를 종합한다."""

    input_paths = inspect_input_paths(
        config=config,
        project_root=project_root,
    )

    pipeline_modules = inspect_pipeline_modules(
        config=config,
    )

    all_files_ready = all(
        item["exists"]
        for item in input_paths
    )

    all_modules_ready = all(
        item["import_status"] == "READY"
        for item in pipeline_modules
    )

    return {
        "experiment_id": config["experiment_id"],
        "config_status": "VALID",
        "input_files_ready": all_files_ready,
        "pipeline_modules_ready": all_modules_ready,
        "ready_for_execution": (
            all_files_ready
            and all_modules_ready
        ),
        "input_paths": input_paths,
        "pipeline_stages": pipeline_modules,
    }


def prepare_output_directory(
    config: dict[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """실험 결과 폴더를 안전하게 준비한다."""

    output_root = resolve_project_path(
        configured_path=(
            config["runtime"]["output_root"]
        ),
        project_root=project_root,
    )

    output_directory = (
        output_root
        / config["experiment_id"]
    )

    if (
        output_directory.exists()
        and any(output_directory.iterdir())
    ):
        raise FileExistsError(
            "동일한 experiment_id의 결과 폴더가 "
            "이미 존재합니다. 기존 결과를 덮어쓰지 않습니다: "
            f"{output_directory}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory


def configure_logger(
    output_directory: Path,
) -> logging.Logger:
    """터미널과 run.log에 동시에 기록하는 Logger를 생성한다."""

    logger = logging.getLogger(
        "experiment_runner"
    )

    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s"
    )

    stream_handler = logging.StreamHandler(
        sys.stdout
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(
        output_directory / "run.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def run_experiment(
    config: dict[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """전체 Experiment Pipeline을 실행한다."""

    output_directory = prepare_output_directory(
        config=config,
        project_root=project_root,
    )

    logger = configure_logger(
        output_directory=output_directory,
    )

    experiment_id = config["experiment_id"]
    entrypoints = config["pipeline"]["entrypoints"]

    logger.info(
        "experiment_started id=%s",
        experiment_id,
    )

    started_at = utc_now()

    context: dict[str, Any] = {}

    try:
        logger.info(
            "pipeline_entrypoints_loading"
        )

        validate_documents = load_callable(
            entrypoints["validate_documents"]
        )
        build_chunks = load_callable(
            entrypoints["build_chunks"]
        )
        resolve_index = load_callable(
            entrypoints["resolve_index"]
        )
        run_batch_retrieval = load_callable(
            entrypoints["run_batch_retrieval"]
        )
        evaluate_experiment = load_callable(
            entrypoints["evaluate_experiment"]
        )
        write_experiment_artifacts = load_callable(
            entrypoints[
                "write_experiment_artifacts"
            ]
        )

        logger.info(
            "pipeline_entrypoints_loaded"
        )

        # ----------------------------------------------------
        # 1. Dataset validation
        # ----------------------------------------------------

        logger.info(
            "stage_started name=validate_documents"
        )

        raw_dataset_info = validate_documents(
            config
        )

        context["dataset_info"] = (
            validate_stage_result(
                stage="validate_documents",
                value=raw_dataset_info,
            )
        )

        logger.info(
            "stage_completed "
            "name=validate_documents "
            "document_count=%s",
            context["dataset_info"][
                "document_count"
            ],
        )

        # ----------------------------------------------------
        # 2. Chunking
        # ----------------------------------------------------

        logger.info(
            "stage_started name=build_chunks"
        )

        raw_chunk_info = build_chunks(
            config,
            context["dataset_info"],
        )

        context["chunk_info"] = (
            validate_stage_result(
                stage="build_chunks",
                value=raw_chunk_info,
            )
        )

        logger.info(
            "stage_completed "
            "name=build_chunks "
            "chunk_count=%s",
            context["chunk_info"]["chunk_count"],
        )

        # ----------------------------------------------------
        # 3. Index resolution
        # ----------------------------------------------------

        logger.info(
            "stage_started name=resolve_index"
        )

        raw_index_info = resolve_index(
            config,
            context["chunk_info"],
        )

        context["index_info"] = (
            validate_stage_result(
                stage="resolve_index",
                value=raw_index_info,
            )
        )

        logger.info(
            "stage_completed "
            "name=resolve_index "
            "index_version=%s "
            "reused=%s",
            context["index_info"][
                "index_version"
            ],
            context["index_info"]["reused"],
        )

        # ----------------------------------------------------
        # 4. Batch retrieval
        # ----------------------------------------------------

        logger.info(
            "stage_started "
            "name=run_batch_retrieval"
        )

        raw_retrieval_results = (
            run_batch_retrieval(
                config,
                context["index_info"],
            )
        )

        context["retrieval_results"] = (
            validate_stage_result(
                stage="run_batch_retrieval",
                value=raw_retrieval_results,
            )
        )

        logger.info(
            "stage_completed "
            "name=run_batch_retrieval "
            "query_count=%s",
            len(context["retrieval_results"]),
        )

        # ----------------------------------------------------
        # 5. Evaluation
        # ----------------------------------------------------

        logger.info(
            "stage_started "
            "name=evaluate_experiment"
        )

        raw_evaluation_result = (
            evaluate_experiment(
                config,
                context["retrieval_results"],
            )
        )

        context["evaluation_result"] = (
            validate_stage_result(
                stage="evaluate_experiment",
                value=raw_evaluation_result,
            )
        )

        logger.info(
            "stage_completed "
            "name=evaluate_experiment"
        )

        # ----------------------------------------------------
        # 6. Artifact writing
        # ----------------------------------------------------

        completed_at = utc_now()

        context["run_metadata"] = {
            "experiment_id": experiment_id,
            "started_at": started_at,
            "completed_at": completed_at,
            "status": "COMPLETED",
        }

        logger.info(
            "stage_started "
            "name=write_experiment_artifacts"
        )

        raw_artifact_info = (
            write_experiment_artifacts(
                config,
                context,
            )
        )

        context["artifact_info"] = (
            validate_stage_result(
                stage=(
                    "write_experiment_artifacts"
                ),
                value=raw_artifact_info,
            )
        )

        logger.info(
            "stage_completed "
            "name=write_experiment_artifacts"
        )

        logger.info(
            "experiment_completed id=%s",
            experiment_id,
        )

        return context

    except Exception:
        logger.exception(
            "experiment_failed id=%s",
            experiment_id,
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    """Experiment Runner CLI 인자를 정의한다."""

    parser = argparse.ArgumentParser(
        description=(
            "공통 YAML 설정을 사용하여 "
            "RAG 검색 실험 전체를 실행합니다."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="실행할 Experiment YAML 경로",
    )

    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Experiment JSON Schema 경로",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="설정만 검증하고 종료",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "실제 모듈을 실행하지 않고 "
            "파일과 모듈 준비 상태만 출력"
        ),
    )

    return parser


def main() -> int:
    """Experiment Runner CLI 진입점."""

    args = build_parser().parse_args()

    config_path = (
        args.config
        if args.config.is_absolute()
        else PROJECT_ROOT / args.config
    )

    schema_path = (
        args.schema
        if args.schema.is_absolute()
        else PROJECT_ROOT / args.schema
    )

    try:
        config = validate_experiment_config(
            config_path=config_path,
            schema_path=schema_path,
            check_files=False,
            project_root=PROJECT_ROOT,
        )

        if args.validate_only:
            print(
                json.dumps(
                    {
                        "experiment_id": (
                            config["experiment_id"]
                        ),
                        "schema_version": (
                            config["schema_version"]
                        ),
                        "status": "VALID",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        if args.dry_run:
            report = build_dry_run_report(
                config=config,
                project_root=PROJECT_ROOT,
            )

            print(
                json.dumps(
                    report,
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        run_experiment(
            config=config,
            project_root=PROJECT_ROOT,
        )

        return 0

    except Exception as exc:
        print(
            f"RUNNER ERROR: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
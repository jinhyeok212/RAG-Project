"""Shared input and output contracts for experiment pipeline stages."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, NotRequired, TypedDict


class ContractError(TypeError):
    """팀 모듈의 반환값이 공통 계약을 위반했을 때 발생하는 예외."""


# ============================================================
# Static type contracts
# ============================================================


class DatasetInfo(TypedDict):
    """validate_documents()가 반환해야 하는 정보."""

    document_path: str
    manifest_path: str
    document_count: int
    sha256: str
    quality_report_path: NotRequired[str]


class ChunkInfo(TypedDict):
    """build_chunks()가 반환해야 하는 정보."""

    chunk_path: str
    manifest_path: str
    chunk_count: int
    chunk_qrels_path: str
    quality_report_path: NotRequired[str]


class IndexInfo(TypedDict):
    """resolve_index()가 반환해야 하는 정보."""

    index_version: str
    db_path: str
    collection_name: str
    reused: bool
    manifest_path: str
    index_count: NotRequired[int]
    build_time_seconds: NotRequired[float]


class RetrievedChunk(TypedDict):
    """질문 한 건에서 검색된 청크 한 개의 형식."""

    rank: int
    chunk_id: str
    document_id: str
    distance: float
    similarity: float
    text: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]


class RetrievalResult(TypedDict):
    """질문 한 건의 검색 결과 형식."""

    query_id: str
    query: str
    retrieval_latency_ms: float
    results: list[RetrievedChunk]
    error: NotRequired[str | None]


class EvaluationResult(TypedDict):
    """evaluate_experiment()가 반환해야 하는 정보."""

    metrics_summary: dict[str, Any]
    metrics_by_query: list[dict[str, Any]]
    failure_cases: list[dict[str, Any]]


class ArtifactInfo(TypedDict):
    """write_experiment_artifacts()가 반환할 수 있는 정보."""

    output_dir: str
    written_files: list[str]


class RunMetadata(TypedDict):
    """Runner 실행 상태 정보."""

    experiment_id: str
    started_at: str
    completed_at: str
    status: str


class ExperimentRunContext(TypedDict):
    """Artifact Writer에 전달하는 전체 실행 결과."""

    dataset_info: DatasetInfo
    chunk_info: ChunkInfo
    index_info: IndexInfo
    retrieval_results: list[RetrievalResult]
    evaluation_result: EvaluationResult
    run_metadata: RunMetadata


# ============================================================
# Common runtime validation helpers
# ============================================================


def require_mapping(
    stage: str,
    value: Any,
) -> dict[str, Any]:
    """값이 Mapping인지 확인하고 일반 dict로 변환한다."""

    if not isinstance(value, Mapping):
        raise ContractError(
            f"{stage} 반환값은 dict여야 합니다. "
            f"현재 타입: {type(value).__name__}"
        )

    return dict(value)


def require_fields(
    stage: str,
    value: Mapping[str, Any],
    required_fields: Sequence[str],
) -> None:
    """필수 필드가 모두 존재하는지 확인한다."""

    missing_fields = [
        field
        for field in required_fields
        if field not in value
    ]

    if missing_fields:
        raise ContractError(
            f"{stage} 반환값에 필수 필드가 없습니다: "
            f"{missing_fields}"
        )


def require_non_empty_string(
    stage: str,
    field_name: str,
    value: Any,
) -> str:
    """필드가 비어 있지 않은 문자열인지 확인한다."""

    if not isinstance(value, str) or not value.strip():
        raise ContractError(
            f"{stage}.{field_name}은 "
            "비어 있지 않은 문자열이어야 합니다."
        )

    return value


def require_non_negative_integer(
    stage: str,
    field_name: str,
    value: Any,
) -> int:
    """필드가 0 이상의 정수인지 확인한다."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ContractError(
            f"{stage}.{field_name}은 "
            "0 이상의 정수여야 합니다."
        )

    return value


def require_positive_integer(
    stage: str,
    field_name: str,
    value: Any,
) -> int:
    """필드가 1 이상의 정수인지 확인한다."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
    ):
        raise ContractError(
            f"{stage}.{field_name}은 "
            "1 이상의 정수여야 합니다."
        )

    return value


def require_non_negative_number(
    stage: str,
    field_name: str,
    value: Any,
) -> float:
    """필드가 0 이상의 유한한 숫자인지 확인한다."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ContractError(
            f"{stage}.{field_name}은 숫자여야 합니다."
        )

    number = float(value)

    if not math.isfinite(number) or number < 0:
        raise ContractError(
            f"{stage}.{field_name}은 "
            "0 이상의 유한한 숫자여야 합니다."
        )

    return number


def require_finite_number(
    stage: str,
    field_name: str,
    value: Any,
) -> float:
    """필드가 NaN이나 Inf가 아닌 숫자인지 확인한다."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ContractError(
            f"{stage}.{field_name}은 숫자여야 합니다."
        )

    number = float(value)

    if not math.isfinite(number):
        raise ContractError(
            f"{stage}.{field_name}은 "
            "유한한 숫자여야 합니다."
        )

    return number


def require_list(
    stage: str,
    field_name: str,
    value: Any,
) -> list[Any]:
    """필드가 list인지 확인한다."""

    if not isinstance(value, list):
        raise ContractError(
            f"{stage}.{field_name}은 list여야 합니다."
        )

    return value


# ============================================================
# Stage result validators
# ============================================================


def validate_dataset_info(
    value: Any,
) -> DatasetInfo:
    """validate_documents() 반환값을 검증한다."""

    stage = "validate_documents"
    result = require_mapping(stage, value)

    require_fields(
        stage=stage,
        value=result,
        required_fields=[
            "document_path",
            "manifest_path",
            "document_count",
            "sha256",
        ],
    )

    require_non_empty_string(
        stage,
        "document_path",
        result["document_path"],
    )
    require_non_empty_string(
        stage,
        "manifest_path",
        result["manifest_path"],
    )
    require_non_negative_integer(
        stage,
        "document_count",
        result["document_count"],
    )
    require_non_empty_string(
        stage,
        "sha256",
        result["sha256"],
    )

    return result  # type: ignore[return-value]


def validate_chunk_info(
    value: Any,
) -> ChunkInfo:
    """build_chunks() 반환값을 검증한다."""

    stage = "build_chunks"
    result = require_mapping(stage, value)

    require_fields(
        stage=stage,
        value=result,
        required_fields=[
            "chunk_path",
            "manifest_path",
            "chunk_count",
            "chunk_qrels_path",
        ],
    )

    require_non_empty_string(
        stage,
        "chunk_path",
        result["chunk_path"],
    )
    require_non_empty_string(
        stage,
        "manifest_path",
        result["manifest_path"],
    )
    require_non_negative_integer(
        stage,
        "chunk_count",
        result["chunk_count"],
    )
    require_non_empty_string(
        stage,
        "chunk_qrels_path",
        result["chunk_qrels_path"],
    )

    return result  # type: ignore[return-value]


def validate_index_info(
    value: Any,
) -> IndexInfo:
    """resolve_index() 반환값을 검증한다."""

    stage = "resolve_index"
    result = require_mapping(stage, value)

    require_fields(
        stage=stage,
        value=result,
        required_fields=[
            "index_version",
            "db_path",
            "collection_name",
            "reused",
            "manifest_path",
        ],
    )

    require_non_empty_string(
        stage,
        "index_version",
        result["index_version"],
    )
    require_non_empty_string(
        stage,
        "db_path",
        result["db_path"],
    )
    require_non_empty_string(
        stage,
        "collection_name",
        result["collection_name"],
    )
    require_non_empty_string(
        stage,
        "manifest_path",
        result["manifest_path"],
    )

    if not isinstance(result["reused"], bool):
        raise ContractError(
            "resolve_index.reused는 Boolean이어야 합니다."
        )

    if "index_count" in result:
        require_non_negative_integer(
            stage,
            "index_count",
            result["index_count"],
        )

    if "build_time_seconds" in result:
        require_non_negative_number(
            stage,
            "build_time_seconds",
            result["build_time_seconds"],
        )

    return result  # type: ignore[return-value]


def validate_retrieved_chunk(
    value: Any,
    query_index: int,
    result_index: int,
) -> RetrievedChunk:
    """검색된 청크 한 건의 형식을 검증한다."""

    stage = (
        "run_batch_retrieval"
        f"[{query_index}].results[{result_index}]"
    )

    result = require_mapping(stage, value)

    require_fields(
        stage=stage,
        value=result,
        required_fields=[
            "rank",
            "chunk_id",
            "document_id",
            "distance",
            "similarity",
        ],
    )

    require_positive_integer(
        stage,
        "rank",
        result["rank"],
    )
    require_non_empty_string(
        stage,
        "chunk_id",
        result["chunk_id"],
    )
    require_non_empty_string(
        stage,
        "document_id",
        result["document_id"],
    )
    require_finite_number(
        stage,
        "distance",
        result["distance"],
    )
    require_finite_number(
        stage,
        "similarity",
        result["similarity"],
    )

    expected_rank = result_index + 1

    if result["rank"] != expected_rank:
        raise ContractError(
            f"{stage}.rank가 검색 순서와 일치하지 않습니다. "
            f"expected={expected_rank}, "
            f"actual={result['rank']}"
        )

    return result  # type: ignore[return-value]


def validate_retrieval_results(
    value: Any,
) -> list[RetrievalResult]:
    """run_batch_retrieval() 전체 반환값을 검증한다."""

    stage = "run_batch_retrieval"

    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
    ):
        raise ContractError(
            f"{stage} 반환값은 list여야 합니다."
        )

    if len(value) == 0:
        raise ContractError(
            f"{stage} 반환값이 비어 있습니다."
        )

    validated_rows: list[RetrievalResult] = []
    query_ids: set[str] = set()

    for query_index, raw_row in enumerate(value):
        row_stage = f"{stage}[{query_index}]"
        row = require_mapping(row_stage, raw_row)

        require_fields(
            stage=row_stage,
            value=row,
            required_fields=[
                "query_id",
                "query",
                "retrieval_latency_ms",
                "results",
            ],
        )

        query_id = require_non_empty_string(
            row_stage,
            "query_id",
            row["query_id"],
        )
        require_non_empty_string(
            row_stage,
            "query",
            row["query"],
        )
        require_non_negative_number(
            row_stage,
            "retrieval_latency_ms",
            row["retrieval_latency_ms"],
        )

        if query_id in query_ids:
            raise ContractError(
                f"중복 query_id가 존재합니다: {query_id}"
            )

        query_ids.add(query_id)

        raw_results = require_list(
            row_stage,
            "results",
            row["results"],
        )

        validated_chunks = []

        for result_index, raw_chunk in enumerate(
            raw_results
        ):
            validated_chunks.append(
                validate_retrieved_chunk(
                    value=raw_chunk,
                    query_index=query_index,
                    result_index=result_index,
                )
            )

        row["results"] = validated_chunks
        validated_rows.append(row)  # type: ignore[arg-type]

    return validated_rows


def validate_evaluation_result(
    value: Any,
) -> EvaluationResult:
    """evaluate_experiment() 반환값을 검증한다."""

    stage = "evaluate_experiment"
    result = require_mapping(stage, value)

    require_fields(
        stage=stage,
        value=result,
        required_fields=[
            "metrics_summary",
            "metrics_by_query",
            "failure_cases",
        ],
    )

    if not isinstance(
        result["metrics_summary"],
        Mapping,
    ):
        raise ContractError(
            "evaluate_experiment.metrics_summary는 "
            "dict여야 합니다."
        )

    require_list(
        stage,
        "metrics_by_query",
        result["metrics_by_query"],
    )
    require_list(
        stage,
        "failure_cases",
        result["failure_cases"],
    )

    return result  # type: ignore[return-value]


def validate_artifact_info(
    value: Any,
) -> ArtifactInfo | None:
    """write_experiment_artifacts() 반환값을 검증한다."""

    if value is None:
        return None

    stage = "write_experiment_artifacts"
    result = require_mapping(stage, value)

    require_fields(
        stage=stage,
        value=result,
        required_fields=[
            "output_dir",
            "written_files",
        ],
    )

    require_non_empty_string(
        stage,
        "output_dir",
        result["output_dir"],
    )

    written_files = require_list(
        stage,
        "written_files",
        result["written_files"],
    )

    for index, written_file in enumerate(written_files):
        require_non_empty_string(
            stage,
            f"written_files[{index}]",
            written_file,
        )

    return result  # type: ignore[return-value]


def validate_stage_result(
    stage: str,
    value: Any,
) -> Any:
    """단계 이름에 맞는 반환값 검증 함수를 실행한다."""

    validators = {
        "validate_documents": validate_dataset_info,
        "build_chunks": validate_chunk_info,
        "resolve_index": validate_index_info,
        "run_batch_retrieval": (
            validate_retrieval_results
        ),
        "evaluate_experiment": (
            validate_evaluation_result
        ),
        "write_experiment_artifacts": (
            validate_artifact_info
        ),
    }

    validator = validators.get(stage)

    if validator is None:
        raise ContractError(
            f"알 수 없는 Pipeline 단계입니다: {stage}"
        )

    return validator(value)
"""Experiment YAML loading and validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "exp_full_kure_v1.yaml"
)

DEFAULT_SCHEMA_PATH = (
    PROJECT_ROOT
    / "schemas"
    / "experiment.schema.json"
)


class ConfigValidationError(ValueError):
    """공통 실험 설정이 규격을 위반했을 때 발생하는 예외."""


def load_yaml(path: Path) -> dict[str, Any]:
    """YAML 파일을 읽어 dict 형태로 반환한다."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Experiment YAML을 찾을 수 없습니다: {path}"
        )

    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ConfigValidationError(
            "Experiment YAML의 최상위 값은 object여야 합니다."
        )

    return config


def load_json_schema(path: Path) -> dict[str, Any]:
    """JSON Schema 파일을 읽어 dict 형태로 반환한다."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Experiment Schema를 찾을 수 없습니다: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as stream:
            schema = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ConfigValidationError(
            f"JSON Schema 문법 오류: {path}:{exc.lineno}:{exc.colno}"
        ) from exc

    if not isinstance(schema, dict):
        raise ConfigValidationError(
            "Experiment Schema의 최상위 값은 object여야 합니다."
        )

    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise ConfigValidationError(
            f"JSON Schema 자체가 올바르지 않습니다: {exc}"
        ) from exc

    return schema


def format_json_path(parts: list[Any]) -> str:
    """JSON Schema 오류 위치를 사람이 읽기 쉬운 경로로 변환한다."""

    if not parts:
        return "$"

    result = "$"

    for part in parts:
        if isinstance(part, int):
            result += f"[{part}]"
        else:
            result += f".{part}"

    return result


def validate_schema(
    config: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    """YAML 설정을 JSON Schema로 검증한다."""

    validator = Draft202012Validator(schema)

    errors = sorted(
        validator.iter_errors(config),
        key=lambda error: list(error.absolute_path),
    )

    if not errors:
        return

    messages = []

    for error in errors:
        location = format_json_path(
            list(error.absolute_path)
        )
        messages.append(
            f"- {location}: {error.message}"
        )

    raise ConfigValidationError(
        "Experiment Schema 검증에 실패했습니다.\n"
        + "\n".join(messages)
    )


def resolve_project_path(
    configured_path: str,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """설정 경로를 프로젝트 내부의 절대 경로로 변환한다."""

    path = Path(configured_path)

    if path.is_absolute():
        raise ConfigValidationError(
            "설정 파일에는 절대 경로를 사용할 수 없습니다: "
            f"{configured_path}"
        )

    root = project_root.resolve()
    resolved = (root / path).resolve()

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ConfigValidationError(
            "설정 경로가 프로젝트 폴더 밖을 가리킵니다: "
            f"{configured_path}"
        ) from exc

    return resolved


def validate_semantics(
    config: dict[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> None:
    """JSON Schema만으로 표현하기 어려운 조건을 검증한다."""

    chunking = config["chunking"]
    retrieval = config["retrieval"]

    chunk_size = chunking["size"]
    chunk_overlap = chunking["overlap"]

    if chunk_overlap >= chunk_size:
        raise ConfigValidationError(
            "chunking.overlap은 chunking.size보다 작아야 합니다. "
            f"현재 값: overlap={chunk_overlap}, size={chunk_size}"
        )

    evaluation_max_k = retrieval["evaluation_max_k"]
    context_top_k = retrieval["context_top_k"]

    if context_top_k > evaluation_max_k:
        raise ConfigValidationError(
            "retrieval.context_top_k는 "
            "retrieval.evaluation_max_k보다 클 수 없습니다. "
            f"현재 값: context_top_k={context_top_k}, "
            f"evaluation_max_k={evaluation_max_k}"
        )

    experiment_id = config["experiment_id"]
    base_experiment_id = config["base_experiment_id"]

    if base_experiment_id == experiment_id:
        raise ConfigValidationError(
            "base_experiment_id는 현재 experiment_id와 "
            "같을 수 없습니다."
        )

    configured_paths = [
        config["dataset"]["document_path"],
        config["dataset"]["chunk_path"],
        config["dataset"]["manifest_path"],
        config["evaluation"]["questions_path"],
        config["evaluation"]["document_qrels_path"],
        config["evaluation"]["chunk_qrels_path"],
        config["evaluation"]["manifest_path"],
        config["index"]["db_path"],
        config["runtime"]["output_root"],
        config["runtime"]["registry_path"],
    ]

    for configured_path in configured_paths:
        resolve_project_path(
            configured_path=configured_path,
            project_root=project_root,
        )


def validate_input_files(
    config: dict[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> None:
    """YAML에 지정한 입력 파일이 실제로 존재하는지 확인한다."""

    input_files = {
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

    missing_files = []

    for field_name, configured_path in input_files.items():
        resolved_path = resolve_project_path(
            configured_path=configured_path,
            project_root=project_root,
        )

        if not resolved_path.is_file():
            missing_files.append(
                f"- {field_name}: {configured_path}"
            )

    if missing_files:
        raise ConfigValidationError(
            "다음 입력 파일이 존재하지 않습니다.\n"
            + "\n".join(missing_files)
        )


def validate_experiment_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    check_files: bool = False,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """YAML을 로드하고 전체 설정 규칙을 검증한다."""

    config = load_yaml(config_path)
    schema = load_json_schema(schema_path)

    validate_schema(
        config=config,
        schema=schema,
    )

    validate_semantics(
        config=config,
        project_root=project_root,
    )

    if check_files:
        validate_input_files(
            config=config,
            project_root=project_root,
        )

    return config


def build_parser() -> argparse.ArgumentParser:
    """CLI 인자를 정의한다."""

    parser = argparse.ArgumentParser(
        description=(
            "공통 Experiment YAML을 "
            "JSON Schema와 추가 규칙으로 검증합니다."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="검증할 Experiment YAML 경로",
    )

    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Experiment JSON Schema 경로",
    )

    parser.add_argument(
        "--check-files",
        action="store_true",
        help="YAML에 지정된 실제 입력 파일 존재 여부도 확인",
    )

    return parser


def main() -> int:
    """Config Validator CLI 진입점."""

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
            check_files=args.check_files,
            project_root=PROJECT_ROOT,
        )
    except Exception as exc:
        print(
            f"CONFIG INVALID: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print("CONFIG VALID")
    print(f"experiment_id: {config['experiment_id']}")
    print(f"schema_version: {config['schema_version']}")

    if args.check_files:
        print("input_files: VALID")
    else:
        print("input_files: NOT CHECKED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Fail-closed, one-run formal TJDR external replication gate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, TypedDict

from nts_med.reproducibility.authorization import claim_one_run_authorization
from nts_med.reproducibility.freeze import sha256_file, validate_freeze_passport
from rps_med.cli.formal_r1 import (
    _git_freeze_valid,
    _interpreter_valid,
    _mapping,
    _relative_file,
    _safe_error_family,
    _write_once,
    _yaml_mapping,
)
from rps_med.development import analyse_source, base_relation_config
from rps_med.external import execute_tjdr_external_gate
from rps_med.tjdr_external_io import load_tjdr_test_measurements
from rps_med.tjdr_io import inspect_tjdr_structure, load_tjdr_train_measurements


class TJDRExternalStage(StrEnum):
    STARTUP_GUARD = "STARTUP_GUARD"
    FREEZE_VALIDATION = "FREEZE_VALIDATION"
    FROZEN_CONTRACT = "FROZEN_CONTRACT"
    SOURCE_PRESENCE = "SOURCE_PRESENCE"
    AUTHORIZATION_CLAIM = "AUTHORIZATION_CLAIM"
    INPUT_IDENTITY = "INPUT_IDENTITY"
    TRAIN_REFERENCE_MEASUREMENT = "TRAIN_REFERENCE_MEASUREMENT"
    TRAIN_REFERENCE_VALIDATION = "TRAIN_REFERENCE_VALIDATION"
    TEST_MEASUREMENT = "TEST_MEASUREMENT"
    BOOTSTRAP_ADJUDICATION = "BOOTSTRAP_ADJUDICATION"
    OUTPUT_PERSISTENCE = "OUTPUT_PERSISTENCE"


class _MeasurementOptions(TypedDict):
    background_threshold: float
    minimum_area_ratio: float
    maximum_area_ratio: float
    substantial_component_ratio: float
    dominant_component_min_share: float
    allow_all_background: bool


@dataclass(slots=True)
class TJDRExternalState:
    current: TJDRExternalStage = TJDRExternalStage.STARTUP_GUARD
    completed: list[TJDRExternalStage] = field(default_factory=list)
    test_scientific_data_may_have_been_opened: bool = False
    source_summaries: dict[str, dict[str, int]] = field(default_factory=dict)

    def enter(self, stage: TJDRExternalStage) -> None:
        self.current = stage

    def complete(self, stage: TJDRExternalStage) -> None:
        if self.current is not stage:
            raise RuntimeError("TJDR external diagnostic stage completion is out of order")
        self.completed.append(stage)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--passport", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--consumption-marker", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-archive", action="append", type=Path, required=True)
    parser.add_argument("--test-archive", action="append", type=Path, required=True)
    parser.add_argument("--development-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _diagnostics(state: TJDRExternalState) -> dict[str, object]:
    return {
        "current_stage": state.current.value,
        "completed_stages": [stage.value for stage in state.completed],
        "test_scientific_data_may_have_been_opened": (
            state.test_scientific_data_may_have_been_opened
        ),
        "source_summaries": dict(state.source_summaries),
        "exception_message_recorded": False,
        "traceback_recorded": False,
        "local_paths_recorded": False,
        "source_identifiers_recorded": False,
    }


def _blocked(reason: str, state: TJDRExternalState) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "BLOCKED_PRE_FORMAL_TJDR_EXTERNAL_GATE",
        "reason_codes": [reason],
        "failed_stage": state.current.value,
        "diagnostics": _diagnostics(state),
        "external_replication_gate": "NOT_RUN",
        "test_scientific_data_opened": False,
        "gate0": "NOT_RERUN",
        "training": "FORBIDDEN",
    }


def _technical_abort(
    error: Exception,
    state: TJDRExternalState,
    *,
    freeze_id: str,
    passport_hash: str,
) -> dict[str, object]:
    family = _safe_error_family(error)
    return {
        "schema_version": 1,
        "status": "FORMAL_RPS_MED_TJDR_EXTERNAL_GATE_ABORTED_TECHNICAL",
        "decision": "ABORTED_TECHNICAL",
        "scientific_result": False,
        "reason_codes": [f"TECHNICAL_ERROR:{state.current.value}:{family}"],
        "failed_stage": state.current.value,
        "error_family": family,
        "freeze_id": freeze_id,
        "freeze_passport_sha256": passport_hash,
        "authorization_consumed": True,
        "diagnostics": _diagnostics(state),
        "external_replication_gate": "NOT_ADJUDICATED",
        "gate0": "NOT_RERUN",
        "automatic_fix": "FORBIDDEN",
        "automatic_retry": "FORBIDDEN",
        "training": "FORBIDDEN",
    }


def _contract(root: Path, passport_path: Path, config_path: Path) -> dict[str, Any]:
    passport = _yaml_mapping(passport_path)
    runner = _mapping(passport.get("runner"), name="runner")
    if runner.get("entry_point") != "rps-med-tjdr-external-gate":
        raise ValueError("TJDR external runner is not frozen")
    artifacts = _mapping(passport.get("artifacts"), name="artifacts")
    record = _mapping(artifacts.get("experiment_config"), name="experiment_config")
    frozen = _relative_file(root, record.get("path"), name="experiment_config")
    if config_path.resolve(strict=False) != frozen:
        raise ValueError("supplied config is not the frozen TJDR external config")
    return _yaml_mapping(config_path)


def _options(config: Mapping[str, Any]) -> _MeasurementOptions:
    annotation = _mapping(config.get("annotation"), name="annotation")
    fov = _mapping(config.get("fov"), name="fov")
    return {
        "background_threshold": float(fov["background_threshold"]),
        "minimum_area_ratio": float(fov["minimum_area_ratio"]),
        "maximum_area_ratio": float(fov["maximum_area_ratio"]),
        "substantial_component_ratio": float(fov["substantial_component_ratio"]),
        "dominant_component_min_share": float(fov["dominant_component_min_share"]),
        "allow_all_background": bool(
            annotation["allow_all_background_complete_annotation"]
        ),
    }


def _identity(
    config: Mapping[str, Any],
    train_archives: Sequence[Path],
    test_archives: Sequence[Path],
    development_reference: Path,
) -> dict[str, object]:
    source = _mapping(config.get("source"), name="source")
    train = inspect_tjdr_structure(train_archives, split="train")
    test = inspect_tjdr_structure(test_archives, split="test")
    checks = {
        "train_archive_sha256": train.archive_sha256
        == tuple(str(value) for value in source["train_archive_sha256"]),
        "test_archive_sha256": test.archive_sha256
        == tuple(str(value) for value in source["test_archive_sha256"]),
        "development_reference_sha256": sha256_file(development_reference)
        == source["development_reference_sha256"],
        "train_count": train.image_count == int(source["expected_development_count"])
        and train.annotation_count == int(source["expected_development_count"]),
        "test_count": test.image_count == int(source["expected_test_count"])
        and test.annotation_count == int(source["expected_test_count"]),
    }
    if not all(checks.values()):
        raise ValueError("TJDR external input identity validation failed")
    return {
        "checks": checks,
        "train_pair_count": train.image_count,
        "test_pair_count": test.image_count,
        "local_paths_recorded": False,
        "source_identifiers_recorded": False,
    }


def _fine_scales(config: Mapping[str, Any]) -> tuple[tuple[float, ...], int]:
    scale = _mapping(config.get("scale"), name="scale")
    minimum = float(scale["log2_minimum"])
    step = float(scale["sensitivity_step"])
    points = int(scale["sensitivity_points"])
    values = tuple(2.0 ** (minimum + index * step) for index in range(points))
    stride = round(float(scale["primary_step"]) / step)
    if len(values) != 33 or stride != 2:
        raise ValueError("TJDR external scale contract is invalid")
    return values, stride


def _development_reference_matches(
    source: object,
    config: Mapping[str, Any],
    reference_path: Path,
) -> bool:
    from nts_med.gate0.execution import SourceMeasurements

    if not isinstance(source, SourceMeasurements):
        return False
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected = reference.get("source")
    scales, stride = _fine_scales(config)
    observed = analyse_source(
        source,
        base_config=base_relation_config(config),
        fine_scales=scales,
        coarse_stride=stride,
        minimum_scale=scales[0],
        maximum_scale=scales[-1],
    )
    return isinstance(expected, dict) and expected == observed


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repository_root.resolve()
    state = TJDRExternalState()
    claimed = False
    passport_hash = ""
    freeze_id = ""
    try:
        state.enter(TJDRExternalStage.STARTUP_GUARD)
        if args.output.exists() or (
            args.consumption_marker is not None and args.consumption_marker.exists()
        ):
            raise FileExistsError("TJDR external output or marker already exists")
        state.complete(TJDRExternalStage.STARTUP_GUARD)

        state.enter(TJDRExternalStage.FREEZE_VALIDATION)
        validation = validate_freeze_passport(root, args.passport, private_manifests={})
        if not validation.valid or validation.freeze_id is None:
            print(json.dumps(_blocked("FREEZE_PASSPORT_VALIDATION_FAILED", state)))
            return 2
        freeze_id = validation.freeze_id
        if not freeze_id.startswith("rps_med_tjdr_external_gate_v0.1_"):
            print(json.dumps(_blocked("FREEZE_ID_NOT_TJDR_EXTERNAL_V0_1", state)))
            return 2
        if not _git_freeze_valid(root, args.passport):
            print(json.dumps(_blocked("GIT_FREEZE_CHECK_FAILED", state)))
            return 2
        if not _interpreter_valid(root):
            print(json.dumps(_blocked("ISOLATED_INTERPRETER_REQUIRED", state)))
            return 2
        state.complete(TJDRExternalStage.FREEZE_VALIDATION)

        state.enter(TJDRExternalStage.FROZEN_CONTRACT)
        config = _contract(root, args.passport, args.config)
        state.complete(TJDRExternalStage.FROZEN_CONTRACT)

        state.enter(TJDRExternalStage.SOURCE_PRESENCE)
        if not all(path.is_file() for path in (*args.train_archive, *args.test_archive)):
            print(json.dumps(_blocked("TJDR_SOURCE_UNAVAILABLE", state)))
            return 2
        if not args.development_reference.is_file():
            print(json.dumps(_blocked("DEVELOPMENT_REFERENCE_UNAVAILABLE", state)))
            return 2
        state.complete(TJDRExternalStage.SOURCE_PRESENCE)

        passport_hash = sha256_file(args.passport)
        if args.mode == "preflight":
            payload = {
                "schema_version": 1,
                "status": "RPS_MED_TJDR_EXTERNAL_PREFLIGHT_PASS_AWAITING_AUTHORIZATION",
                "freeze_id": freeze_id,
                "freeze_passport_sha256": passport_hash,
                "raw_input_bytes_opened": False,
                "test_scientific_data_opened": False,
                "authorization_claimed": False,
                "external_replication_gate": "NOT_RUN",
                "gate0": "NOT_RERUN",
                "training": "FORBIDDEN",
                "diagnostics": _diagnostics(state),
            }
            _write_once(args.output, payload)
            print(json.dumps({"status": payload["status"]}))
            return 0

        if args.authorization is None or args.consumption_marker is None:
            print(json.dumps(_blocked("RUN_REQUIRES_NEW_AUTHORIZATION", state)))
            return 2
        state.enter(TJDRExternalStage.AUTHORIZATION_CLAIM)
        claim = claim_one_run_authorization(
            args.authorization,
            args.consumption_marker,
            passport_sha256=passport_hash,
            freeze_id=freeze_id,
            authorization_scope="FORMAL_TJDR_EXTERNAL_GATE_ONCE",
        )
        claimed = True
        state.complete(TJDRExternalStage.AUTHORIZATION_CLAIM)

        state.enter(TJDRExternalStage.INPUT_IDENTITY)
        input_identity = _identity(
            config,
            args.train_archive,
            args.test_archive,
            args.development_reference,
        )
        state.complete(TJDRExternalStage.INPUT_IDENTITY)

        source_config = _mapping(config.get("source"), name="source")
        options = _options(config)
        state.enter(TJDRExternalStage.TRAIN_REFERENCE_MEASUREMENT)
        train = load_tjdr_train_measurements(
            args.train_archive,
            expected_count=int(source_config["expected_development_count"]),
            **options,
        )
        state.source_summaries["tjdr_official_train"] = {
            "expected": train.expected_input_count,
            "eligible": len(train.records),
            "excluded": train.measurement_exclusion_count,
        }
        state.complete(TJDRExternalStage.TRAIN_REFERENCE_MEASUREMENT)

        state.enter(TJDRExternalStage.TRAIN_REFERENCE_VALIDATION)
        if not _development_reference_matches(train, config, args.development_reference):
            raise ValueError("TJDR train reference does not reproduce exactly")
        state.complete(TJDRExternalStage.TRAIN_REFERENCE_VALIDATION)

        state.enter(TJDRExternalStage.TEST_MEASUREMENT)
        state.test_scientific_data_may_have_been_opened = True
        test = load_tjdr_test_measurements(
            args.test_archive,
            expected_count=int(source_config["expected_test_count"]),
            **options,
        )
        state.source_summaries["tjdr_official_test"] = {
            "expected": test.expected_input_count,
            "eligible": len(test.records),
            "excluded": test.measurement_exclusion_count,
        }
        state.complete(TJDRExternalStage.TEST_MEASUREMENT)

        state.enter(TJDRExternalStage.BOOTSTRAP_ADJUDICATION)
        result = execute_tjdr_external_gate(train, test, config)
        state.complete(TJDRExternalStage.BOOTSTRAP_ADJUDICATION)
        result.update(
            {
                "freeze_id": freeze_id,
                "freeze_passport_sha256": passport_hash,
                "authorization_sha256": claim.authorization_sha256,
                "authorization_marker_sha256": claim.marker_sha256,
                "authorization_consumed": True,
                "input_identity": input_identity,
                "test_opened_after_authorization_claim": True,
                "diagnostics": _diagnostics(state),
                "raw_data_persisted": False,
                "model_weights_persisted": False,
            }
        )
        state.enter(TJDRExternalStage.OUTPUT_PERSISTENCE)
        _write_once(args.output, result)
        print(json.dumps({"status": result["status"], "decision": result["decision"]}))
        return 0
    except Exception as error:
        authorization_consumed = claimed or (
            state.current is TJDRExternalStage.AUTHORIZATION_CLAIM
            and args.consumption_marker is not None
            and args.consumption_marker.exists()
        )
        if not authorization_consumed:
            print(
                json.dumps(
                    _blocked(
                        f"TECHNICAL_ERROR:{state.current.value}:{_safe_error_family(error)}",
                        state,
                    )
                )
            )
            return 2
        payload = _technical_abort(
            error, state, freeze_id=freeze_id, passport_hash=passport_hash
        )
        with suppress(FileExistsError, OSError, ValueError):
            _write_once(args.output, payload)
        print(json.dumps({"status": payload["status"], "failed_stage": state.current.value}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

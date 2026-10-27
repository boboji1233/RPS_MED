"""Fail-closed, one-authorization formal RPS-Med R1 entry point."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from nts_med.reproducibility.authorization import claim_one_run_authorization
from nts_med.reproducibility.freeze import (
    manifest_arguments,
    protected_paths,
    sha256_file,
    validate_freeze_passport,
)
from rps_med.formal import execute_formal_r1
from rps_med.source_io import (
    inspect_development_boundary,
    load_ddr_confirmation_measurements,
    load_maples_confirmation_measurements,
    validate_ddr_confirmation_manifest,
)


class R1Stage(StrEnum):
    """Stable, privacy-safe formal R1 execution stages."""

    STARTUP_GUARD = "STARTUP_GUARD"
    FREEZE_VALIDATION = "FREEZE_VALIDATION"
    FROZEN_CONTRACT = "FROZEN_CONTRACT"
    SOURCE_PRESENCE = "SOURCE_PRESENCE"
    AUTHORIZATION_CLAIM = "AUTHORIZATION_CLAIM"
    INPUT_IDENTITY = "INPUT_IDENTITY"
    MAPLES_CONFIRMATION_MEASUREMENT = "MAPLES_CONFIRMATION_MEASUREMENT"
    DDR_CONFIRMATION_MEASUREMENT = "DDR_CONFIRMATION_MEASUREMENT"
    BOOTSTRAP_ADJUDICATION = "BOOTSTRAP_ADJUDICATION"
    OUTPUT_PERSISTENCE = "OUTPUT_PERSISTENCE"


@dataclass(slots=True)
class R1DiagnosticState:
    current: R1Stage = R1Stage.STARTUP_GUARD
    completed: list[R1Stage] = field(default_factory=list)
    scientific_data_may_have_been_opened: bool = False
    source_summaries: dict[str, dict[str, int]] = field(default_factory=dict)

    def enter(self, stage: R1Stage) -> None:
        self.current = stage

    def complete(self, stage: R1Stage) -> None:
        if self.current is not stage:
            raise RuntimeError("R1 diagnostic stage completion is out of order")
        self.completed.append(stage)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--passport", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--consumption-marker", type=Path)
    parser.add_argument(
        "--private-manifest", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--maples-archive", type=Path, required=True)
    parser.add_argument("--maples-additional", type=Path, required=True)
    parser.add_argument("--ddr-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _yaml_mapping(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise ValueError("YAML input must be a string-keyed mapping")
    return loaded


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return value


def _relative_file(root: Path, raw: object, *, name: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{name} path is invalid")
    portable = PurePosixPath(raw)
    if portable.is_absolute() or ".." in portable.parts:
        raise ValueError(f"{name} path is invalid")
    path = (root / Path(*portable.parts)).resolve(strict=False)
    path.relative_to(root.resolve())
    if not path.is_file():
        raise ValueError(f"{name} file is unavailable")
    return path


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _git_freeze_valid(root: Path, passport: Path) -> bool:
    payload = _yaml_mapping(passport)
    commit = payload.get("implementation_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        return False
    if _git(root, "cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        return False
    if _git(root, "merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
        return False
    paths = protected_paths(passport)
    dirty = _git(root, "status", "--porcelain", "--", *paths)
    return dirty.returncode == 0 and not dirty.stdout.strip()


def _interpreter_valid(root: Path) -> bool:
    expected = (root / ".conda" / "nts-med" / "python.exe").resolve(strict=False)
    return Path(sys.executable).resolve(strict=False) == expected


def _diagnostics(state: R1DiagnosticState) -> dict[str, object]:
    return {
        "current_stage": state.current.value,
        "completed_stages": [stage.value for stage in state.completed],
        "scientific_data_may_have_been_opened": state.scientific_data_may_have_been_opened,
        "source_summaries": dict(state.source_summaries),
        "exception_message_recorded": False,
        "traceback_recorded": False,
        "local_paths_recorded": False,
        "source_identifiers_recorded": False,
    }


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
    except FileExistsError:
        raise


def _blocked(reason: str, state: R1DiagnosticState) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "BLOCKED_PRE_FORMAL_R1",
        "reason_codes": [reason],
        "failed_stage": state.current.value,
        "diagnostics": _diagnostics(state),
        "formal_r1": "NOT_RUN",
        "scientific_data_opened": False,
        "gate0": "NOT_RERUN",
        "training": "FORBIDDEN",
    }


def _safe_error_family(error: Exception) -> str:
    allowed = {
        FileExistsError: "FILE_EXISTS_ERROR",
        KeyError: "KEY_ERROR",
        OSError: "OS_ERROR",
        PermissionError: "PERMISSION_ERROR",
        RuntimeError: "RUNTIME_ERROR",
        TypeError: "TYPE_ERROR",
        ValueError: "VALUE_ERROR",
    }
    return next(
        (name for error_type, name in allowed.items() if isinstance(error, error_type)),
        "UNEXPECTED_ERROR",
    )


def _load_contract(
    root: Path, passport_path: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    passport = _yaml_mapping(passport_path)
    runner = _mapping(passport.get("runner"), name="runner")
    if runner.get("entry_point") != "rps-med-formal-r1":
        raise ValueError("formal R1 runner is not frozen")
    artifacts = _mapping(passport.get("artifacts"), name="artifacts")
    config_record = _mapping(artifacts.get("experiment_config"), name="experiment_config")
    frozen_config = _relative_file(
        root, config_record.get("path"), name="experiment_config"
    )
    if config_path.resolve(strict=False) != frozen_config:
        raise ValueError("supplied config is not the frozen formal R1 config")
    config = _yaml_mapping(config_path)
    if config.get("config_id") != "rps_med_r1_v0.5":
        raise ValueError("formal R1 config identity is invalid")
    execution = _mapping(config.get("execution"), name="execution")
    if execution.get("formal_r1") != "NOT_AUTHORIZED":
        raise ValueError("formal R1 config must await authorization")
    if execution.get("model_training") != "FORBIDDEN":
        raise ValueError("formal R1 config must forbid training")
    return passport, config


def _verify_identity(
    config: Mapping[str, Any],
    *,
    maples_archive: Path,
    maples_additional: Path,
    ddr_root: Path,
    ddr_manifest: Path,
    root: Path,
) -> dict[str, object]:
    identity = _mapping(config.get("input_identity"), name="input_identity")
    matches = {
        "maples_main_archive": sha256_file(maples_archive)
        == identity.get("maples_main_archive"),
        "maples_additional_archive": sha256_file(maples_additional)
        == identity.get("maples_additional_archive"),
        "ddr_private_manifest": sha256_file(ddr_manifest)
        == identity.get("ddr_private_manifest"),
    }
    boundary_path = _relative_file(
        root, identity.get("boundary_lock"), name="boundary_lock"
    )
    boundary_lock = _yaml_mapping(boundary_path)
    expected_boundary = _mapping(
        boundary_lock.get("development_boundary"), name="development_boundary"
    )
    boundary_config = _mapping(
        config.get("confirmation_boundary"), name="confirmation_boundary"
    )
    boundary = inspect_development_boundary(
        maples_archive,
        ddr_root,
        maples_development_count=int(boundary_config["maples_development_count"]),
    )
    actual_boundary = {
        field: getattr(boundary, field) for field in expected_boundary
    }
    matches["development_confirmation_boundary"] = (
        dict(expected_boundary) == actual_boundary
    )
    subset = validate_ddr_confirmation_manifest(ddr_root, ddr_manifest)
    matches["ddr_confirmation_manifest_subset"] = subset.get("valid") is True
    if not all(matches.values()):
        raise ValueError("formal R1 input identity validation failed")
    return {
        "checks": matches,
        "ddr_confirmation_expected_file_count": subset["expected_file_count"],
        "ddr_confirmation_observed_file_count": subset["observed_file_count"],
        "local_paths_recorded": False,
        "source_identifiers_recorded": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repository_root.resolve()
    state = R1DiagnosticState()
    claimed = False
    passport_hash = ""
    freeze_id = ""
    try:
        state.enter(R1Stage.STARTUP_GUARD)
        if args.output.exists() or (
            args.consumption_marker is not None and args.consumption_marker.exists()
        ):
            raise FileExistsError("formal R1 output or marker already exists")
        private_manifests = manifest_arguments(args.private_manifest)
        if set(private_manifests) != {"ddr"}:
            raise ValueError("formal R1 requires exactly the DDR private manifest")
        state.complete(R1Stage.STARTUP_GUARD)

        state.enter(R1Stage.FREEZE_VALIDATION)
        validation = validate_freeze_passport(
            root, args.passport, private_manifests=private_manifests
        )
        if not validation.valid or validation.freeze_id is None:
            print(json.dumps(_blocked("FREEZE_PASSPORT_VALIDATION_FAILED", state)))
            return 2
        freeze_id = validation.freeze_id
        if not freeze_id.startswith("rps_med_r1_v0.5_"):
            print(json.dumps(_blocked("FREEZE_ID_NOT_RPS_R1_V0_5", state)))
            return 2
        if not _git_freeze_valid(root, args.passport):
            print(json.dumps(_blocked("GIT_FREEZE_CHECK_FAILED", state)))
            return 2
        if not _interpreter_valid(root):
            print(json.dumps(_blocked("ISOLATED_INTERPRETER_REQUIRED", state)))
            return 2
        state.complete(R1Stage.FREEZE_VALIDATION)

        state.enter(R1Stage.FROZEN_CONTRACT)
        _, config = _load_contract(root, args.passport, args.config)
        state.complete(R1Stage.FROZEN_CONTRACT)

        state.enter(R1Stage.SOURCE_PRESENCE)
        if not args.maples_archive.is_file() or not args.maples_additional.is_file():
            print(json.dumps(_blocked("MAPLES_SOURCE_UNAVAILABLE", state)))
            return 2
        if not args.ddr_root.is_dir():
            print(json.dumps(_blocked("DDR_SOURCE_UNAVAILABLE", state)))
            return 2
        state.complete(R1Stage.SOURCE_PRESENCE)

        passport_hash = sha256_file(args.passport)
        if args.mode == "preflight":
            payload = {
                "schema_version": 1,
                "status": "RPS_MED_R1_V0_5_PREFLIGHT_PASS_AWAITING_EXACT_HASH_AUTHORIZATION",
                "freeze_id": freeze_id,
                "freeze_passport_sha256": passport_hash,
                "scientific_data_opened": False,
                "raw_input_bytes_opened": False,
                "authorization_claimed": False,
                "formal_r1": "NOT_RUN",
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

        state.enter(R1Stage.AUTHORIZATION_CLAIM)
        claim = claim_one_run_authorization(
            args.authorization,
            args.consumption_marker,
            passport_sha256=passport_hash,
            freeze_id=freeze_id,
            authorization_scope="FORMAL_R1_ONCE",
        )
        claimed = True
        state.complete(R1Stage.AUTHORIZATION_CLAIM)

        state.enter(R1Stage.INPUT_IDENTITY)
        input_identity = _verify_identity(
            config,
            maples_archive=args.maples_archive,
            maples_additional=args.maples_additional,
            ddr_root=args.ddr_root,
            ddr_manifest=private_manifests["ddr"],
            root=root,
        )
        state.complete(R1Stage.INPUT_IDENTITY)

        boundary = _mapping(
            config.get("confirmation_boundary"), name="confirmation_boundary"
        )
        state.enter(R1Stage.MAPLES_CONFIRMATION_MEASUREMENT)
        state.scientific_data_may_have_been_opened = True
        maples = load_maples_confirmation_measurements(
            args.maples_archive,
            args.maples_additional,
            development_count=int(boundary["maples_development_count"]),
        )
        if maples.expected_input_count != int(boundary["maples_confirmation_count"]):
            raise ValueError("MAPLES confirmation count differs from frozen boundary")
        state.source_summaries["maples_dr_messidor1"] = {
            "expected": maples.expected_input_count,
            "eligible": len(maples.records),
            "excluded": maples.measurement_exclusion_count,
        }
        state.complete(R1Stage.MAPLES_CONFIRMATION_MEASUREMENT)

        state.enter(R1Stage.DDR_CONFIRMATION_MEASUREMENT)
        ddr = load_ddr_confirmation_measurements(args.ddr_root)
        if ddr.expected_input_count != int(boundary["ddr_confirmation_count"]):
            raise ValueError("DDR confirmation count differs from frozen boundary")
        state.source_summaries["ddr"] = {
            "expected": ddr.expected_input_count,
            "eligible": len(ddr.records),
            "excluded": ddr.measurement_exclusion_count,
        }
        state.complete(R1Stage.DDR_CONFIRMATION_MEASUREMENT)

        state.enter(R1Stage.BOOTSTRAP_ADJUDICATION)
        result = execute_formal_r1(maples, ddr, config)
        state.complete(R1Stage.BOOTSTRAP_ADJUDICATION)
        result.update(
            {
                "freeze_id": freeze_id,
                "freeze_passport_sha256": passport_hash,
                "authorization_sha256": claim.authorization_sha256,
                "authorization_marker_sha256": claim.marker_sha256,
                "authorization_consumed": True,
                "input_identity": input_identity,
                "confirmation_opened_after_authorization_claim": True,
                "diagnostics": _diagnostics(state),
                "raw_data_persisted": False,
                "model_weights_persisted": False,
            }
        )
        state.enter(R1Stage.OUTPUT_PERSISTENCE)
        _write_once(args.output, result)
        print(json.dumps({"status": result["status"], "decision": result["decision"]}))
        return 0
    except Exception as error:
        authorization_consumed = claimed or (
            state.current is R1Stage.AUTHORIZATION_CLAIM
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
        payload = {
            "schema_version": 1,
            "status": "FORMAL_RPS_MED_R1_ABORTED_TECHNICAL",
            "decision": "ABORTED_TECHNICAL",
            "scientific_result": False,
            "reason_codes": [
                f"TECHNICAL_ERROR:{state.current.value}:{_safe_error_family(error)}"
            ],
            "failed_stage": state.current.value,
            "error_family": _safe_error_family(error),
            "freeze_id": freeze_id,
            "freeze_passport_sha256": passport_hash,
            "authorization_consumed": True,
            "diagnostics": _diagnostics(state),
            "formal_r1": "NOT_ADJUDICATED",
            "gate0": "NOT_RERUN",
            "automatic_retry": "FORBIDDEN",
            "training": "FORBIDDEN",
        }
        with suppress(FileExistsError, OSError, ValueError):
            _write_once(args.output, payload)
        print(json.dumps({"status": payload["status"], "failed_stage": state.current.value}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

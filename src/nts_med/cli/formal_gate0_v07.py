"""Fail-closed Gate 0 v0.7 entry point with privacy-safe stage diagnostics.

Version 0.7 is a technical-diagnostic amendment only.  It reuses the frozen
v0.6 source measurement and aggregate scientific executor without changing
their scientific behavior.
"""

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

from nts_med.gate0.execution import SourceMeasurements, execute_gate0_v06
from nts_med.gate0.source_io import (
    load_ddr_test_measurements,
    load_maples_test_measurements,
)
from nts_med.reproducibility.authorization import claim_one_run_authorization
from nts_med.reproducibility.freeze import (
    manifest_arguments,
    protected_paths,
    sha256_file,
    validate_freeze_passport,
)
from nts_med.reproducibility.input_identity import validate_directory_manifest


class InputIdentityError(ValueError):
    """A frozen source no longer matches its bound local identity."""


class TechnicalStage(StrEnum):
    """Stable, privacy-safe stages admitted to formal v0.7 output."""

    STARTUP_GUARD = "STARTUP_GUARD"
    FREEZE_VALIDATION = "FREEZE_VALIDATION"
    FROZEN_CONTRACT = "FROZEN_CONTRACT"
    SOURCE_PRESENCE = "SOURCE_PRESENCE"
    AUTHORIZATION_CLAIM = "AUTHORIZATION_CLAIM"
    INPUT_IDENTITY = "INPUT_IDENTITY"
    MAPLES_SOURCE_MEASUREMENT = "MAPLES_SOURCE_MEASUREMENT"
    DDR_SOURCE_MEASUREMENT = "DDR_SOURCE_MEASUREMENT"
    AGGREGATE_EXECUTION = "AGGREGATE_EXECUTION"
    OUTPUT_PERSISTENCE = "OUTPUT_PERSISTENCE"


@dataclass(slots=True)
class DiagnosticState:
    """In-memory stage trace containing no paths or source identifiers."""

    current: TechnicalStage = TechnicalStage.STARTUP_GUARD
    completed: list[TechnicalStage] = field(default_factory=list)
    scientific_data_may_have_been_opened: bool = False
    source_summaries: dict[str, dict[str, object]] = field(default_factory=dict)

    def enter(self, stage: TechnicalStage) -> None:
        self.current = stage

    def complete(self, stage: TechnicalStage) -> None:
        if self.current is not stage:
            raise RuntimeError("diagnostic stage completion is out of order")
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
    parser.add_argument("--maples-variability", type=Path, required=True)
    parser.add_argument("--ddr-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _yaml_mapping(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise ValueError("YAML input must be a string-keyed mapping")
    return loaded


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


def _relative_repository_file(root: Path, raw: object, *, name: str) -> Path:
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


def _load_v07_config(root: Path, overlay_path: Path) -> dict[str, Any]:
    overlay = _yaml_mapping(overlay_path)
    allowed = {
        "schema_version",
        "name",
        "preregistration",
        "execution_status",
        "gate0_permitted",
        "training_permitted",
        "reuse_prior_passport_or_authorization",
        "scientific_base_config",
        "scientific_base_config_sha256",
        "diagnostics",
        "implementation",
    }
    if set(overlay) - allowed:
        raise ValueError("v0.7 overlay contains non-diagnostic fields")
    if overlay.get("schema_version") != 1 or overlay.get("name") != "gate0_dr_v0_7":
        raise ValueError("v0.7 overlay identity is invalid")
    if overlay.get("preregistration") != "metadata/preregistration/gate0_dr_v0.7.yaml":
        raise ValueError("v0.7 preregistration binding is invalid")
    if overlay.get("gate0_permitted") is not False:
        raise ValueError("v0.7 Gate 0 must remain unauthorized before exact-hash approval")
    if overlay.get("training_permitted") is not False:
        raise ValueError("v0.7 training must remain forbidden")
    if overlay.get("reuse_prior_passport_or_authorization") is not False:
        raise ValueError("prior passports and authorizations must not be reusable")

    base_relative = overlay.get("scientific_base_config")
    if base_relative != "configs/experiment/gate0_dr_v0.6.yaml":
        raise ValueError("v0.7 scientific base must be the frozen v0.6 config")
    base_path = _relative_repository_file(
        root, base_relative, name="scientific_base_config"
    )
    expected_hash = overlay.get("scientific_base_config_sha256")
    if not isinstance(expected_hash, str) or sha256_file(base_path) != expected_hash:
        raise ValueError("v0.7 scientific base config hash mismatch")

    diagnostics = overlay.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("v0.7 diagnostics configuration is unavailable")
    required_diagnostics = {
        "privacy_safe_stage_codes": True,
        "stable_reason_codes": True,
        "aggregate_source_counts_only": True,
        "exception_messages_in_output": False,
        "tracebacks_in_output": False,
        "scientific_parameter_changes": False,
    }
    if dict(diagnostics) != required_diagnostics:
        raise ValueError("v0.7 diagnostics configuration is invalid")

    base = _yaml_mapping(base_path)
    merged = dict(base)
    merged.update(
        {
            "name": overlay["name"],
            "preregistration": overlay["preregistration"],
            "execution_status": overlay.get("execution_status"),
            "gate0_permitted": False,
            "training_permitted": False,
            "reuse_prior_passport_or_authorization": False,
            "diagnostics": dict(diagnostics),
        }
    )
    implementation = dict(
        _mapping(overlay.get("implementation"), name="implementation")
    )
    merged["implementation"] = implementation
    return merged


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return value


def _v07_passport_contract(
    root: Path,
    passport_path: Path,
    config_path: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    passport = _yaml_mapping(passport_path)
    runner = passport.get("runner")
    if not isinstance(runner, Mapping) or runner.get("entry_point") != (
        "nts-med-formal-gate0-v07"
    ):
        raise ValueError("v0.7 formal runner is not frozen")
    artifacts = passport.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("freeze artifacts are unavailable")
    raw_config = artifacts.get("experiment_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("frozen v0.7 experiment config is unavailable")
    relative = raw_config.get("path")
    if relative != "configs/experiment/gate0_dr_v0.7.yaml":
        raise ValueError("frozen v0.7 experiment config path is invalid")
    frozen_config = _relative_repository_file(root, relative, name="experiment_config")
    if config_path.resolve(strict=False) != frozen_config:
        raise ValueError("the supplied config is not the frozen v0.7 config")

    config = _load_v07_config(root, config_path)
    raw_identity = config.get("input_identity")
    if not isinstance(raw_identity, Mapping):
        raise ValueError("frozen input identity configuration is unavailable")
    declared_archives = raw_identity.get("maples_archives")
    if not isinstance(declared_archives, Mapping):
        raise ValueError("frozen archive identities are unavailable")
    raw_input_files = passport.get("input_files")
    if not isinstance(raw_input_files, Mapping) or set(raw_input_files) != {
        "maples_main_archive",
        "maples_additional_archive",
        "maples_variability_archive",
    }:
        raise ValueError("v0.7 archive identities are incomplete")
    if not all(isinstance(value, str) for value in raw_input_files.values()):
        raise ValueError("v0.7 archive identities are invalid")
    if dict(raw_input_files) != dict(declared_archives):
        raise ValueError("passport archives do not match the frozen scientific base")
    raw_manifests = passport.get("input_manifests")
    ddr_digest = raw_identity.get("ddr_private_manifest_sha256")
    if raw_manifests != {"ddr": ddr_digest}:
        raise ValueError("passport DDR manifest does not match the frozen scientific base")
    return (
        {str(name): str(value) for name, value in raw_input_files.items()},
        config,
    )


def _verify_bound_inputs(
    archive_hashes: Mapping[str, str],
    private_manifests: Mapping[str, Path],
    *,
    maples_archive: Path,
    maples_additional: Path,
    maples_variability: Path,
    ddr_root: Path,
) -> dict[str, object]:
    archives = {
        "maples_main_archive": maples_archive,
        "maples_additional_archive": maples_additional,
        "maples_variability_archive": maples_variability,
    }
    if any(sha256_file(path) != archive_hashes[name] for name, path in archives.items()):
        raise InputIdentityError("MAPLES archive identity mismatch")
    if set(private_manifests) != {"ddr"}:
        raise InputIdentityError("DDR private manifest is required")
    ddr = validate_directory_manifest(ddr_root, private_manifests["ddr"])
    if not ddr.valid:
        raise InputIdentityError("DDR directory identity mismatch")
    return {
        "maples_archive_count": len(archives),
        "maples_archives_sha256_verified": True,
        "ddr_expected_file_count": ddr.expected_file_count,
        "ddr_observed_file_count": ddr.observed_file_count,
        "ddr_private_manifest_verified": True,
        "raw_paths_recorded": False,
        "source_identifiers_recorded": False,
    }


def _bounds(config: Mapping[str, Any], source: str) -> tuple[float, float]:
    measurement = config.get("measurement")
    if not isinstance(measurement, Mapping):
        raise ValueError("measurement configuration is unavailable")
    extreme = measurement.get("extreme_fov_ratio_sensitivity")
    if not isinstance(extreme, Mapping):
        raise ValueError("extreme-FOV configuration is unavailable")
    values = extreme.get(source)
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError("source FOV bounds are invalid")
    lower, upper = (float(value) for value in values)
    if not 0.0 < lower <= upper <= 1.0:
        raise ValueError("source FOV bounds are invalid")
    return lower, upper


def _source_summary(source: SourceMeasurements) -> dict[str, object]:
    return {
        "expected_input_count": source.expected_input_count,
        "observed_input_count": source.observed_input_count,
        "measurement_eligible_count": len(source.records),
        "measurement_exclusion_count": source.measurement_exclusion_count,
        "reader_measurements_present": source.reader is not None,
        "source_identifiers_recorded": False,
    }


def _safe_error_family(error: Exception) -> str:
    if isinstance(error, InputIdentityError):
        return "INPUT_IDENTITY_ERROR"
    if isinstance(error, PermissionError):
        return "PERMISSION_ERROR"
    if isinstance(error, FileExistsError):
        return "OUTPUT_EXISTS_ERROR"
    if isinstance(error, yaml.YAMLError):
        return "YAML_ERROR"
    if isinstance(error, TypeError):
        return "TYPE_ERROR"
    if isinstance(error, ValueError):
        return "VALUE_ERROR"
    if isinstance(error, OSError):
        return "IO_ERROR"
    return "UNEXPECTED_ERROR"


def _stage_diagnostics(state: DiagnosticState) -> dict[str, object]:
    return {
        "completed_stages": [stage.value for stage in state.completed],
        "last_completed_stage": state.completed[-1].value if state.completed else None,
        "source_measurement_summaries": dict(sorted(state.source_summaries.items())),
        "exception_message_recorded": False,
        "traceback_recorded": False,
        "absolute_paths_recorded": False,
        "source_identifiers_recorded": False,
    }


def _write_once(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError("formal output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _blocked(reason: str, state: DiagnosticState) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "BLOCKED_PRE_GATE0_V07",
        "reason_codes": [reason],
        "failed_stage": state.current.value,
        "diagnostics": _stage_diagnostics(state),
        "gate0": "NOT_RUN",
        "scientific_data_opened": False,
        "training": "FORBIDDEN",
    }


def _technical_abort(
    error: Exception,
    state: DiagnosticState,
    *,
    freeze_id: str,
    passport_hash: str,
) -> dict[str, object]:
    family = _safe_error_family(error)
    return {
        "schema_version": 1,
        "status": "FORMAL_GATE0_V07_ABORTED_TECHNICAL",
        "decision": "ABORTED_TECHNICAL",
        "scientific_result": False,
        "reason_codes": [f"TECHNICAL_ERROR:{state.current.value}:{family}"],
        "failed_stage": state.current.value,
        "error_family": family,
        "freeze_id": freeze_id,
        "freeze_passport_sha256": passport_hash,
        "authorization_consumed": True,
        "scientific_data_may_have_been_opened": (
            state.scientific_data_may_have_been_opened
        ),
        "diagnostics": _stage_diagnostics(state),
        "gate0": "NOT_ADJUDICATED",
        "rerun_without_new_version_and_authorization": "FORBIDDEN",
        "training": "FORBIDDEN",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repository_root.resolve()
    state = DiagnosticState()
    claimed = False
    passport_hash = ""
    freeze_id = ""
    try:
        state.enter(TechnicalStage.STARTUP_GUARD)
        if args.output.exists() or (
            args.consumption_marker is not None and args.consumption_marker.exists()
        ):
            raise FileExistsError("formal output or consumption marker already exists")
        private_manifests = manifest_arguments(args.private_manifest)
        state.complete(TechnicalStage.STARTUP_GUARD)

        state.enter(TechnicalStage.FREEZE_VALIDATION)
        validation = validate_freeze_passport(
            root, args.passport, private_manifests=private_manifests
        )
        if not validation.valid or validation.freeze_id is None:
            payload = _blocked("FREEZE_PASSPORT_VALIDATION_FAILED", state)
            print(json.dumps(payload, sort_keys=True))
            return 2
        freeze_id = validation.freeze_id
        if not freeze_id.startswith("gate0_dr_v0.7_"):
            payload = _blocked("FREEZE_ID_NOT_V0_7", state)
            print(json.dumps(payload, sort_keys=True))
            return 2
        if not _git_freeze_valid(root, args.passport):
            payload = _blocked("GIT_FREEZE_CHECK_FAILED", state)
            print(json.dumps(payload, sort_keys=True))
            return 2
        if not _interpreter_valid(root):
            payload = _blocked("ISOLATED_INTERPRETER_REQUIRED", state)
            print(json.dumps(payload, sort_keys=True))
            return 2
        state.complete(TechnicalStage.FREEZE_VALIDATION)

        state.enter(TechnicalStage.FROZEN_CONTRACT)
        archive_hashes, config = _v07_passport_contract(
            root, args.passport, args.config
        )
        maples_bounds = _bounds(config, "maples_dr_messidor1")
        ddr_bounds = _bounds(config, "ddr")
        state.complete(TechnicalStage.FROZEN_CONTRACT)

        state.enter(TechnicalStage.SOURCE_PRESENCE)
        for path in (
            args.maples_archive,
            args.maples_additional,
            args.maples_variability,
        ):
            if not path.is_file():
                payload = _blocked("REQUIRED_SOURCE_INPUT_UNAVAILABLE", state)
                print(json.dumps(payload, sort_keys=True))
                return 2
        if not args.ddr_root.is_dir():
            payload = _blocked("REQUIRED_SOURCE_INPUT_UNAVAILABLE", state)
            print(json.dumps(payload, sort_keys=True))
            return 2
        state.complete(TechnicalStage.SOURCE_PRESENCE)

        passport_hash = sha256_file(args.passport)
        if args.mode == "preflight":
            payload = {
                "schema_version": 1,
                "status": "V0_7_PREFLIGHT_PASS_AWAITING_EXACT_HASH_AUTHORIZATION",
                "freeze_id": freeze_id,
                "freeze_passport_sha256": passport_hash,
                "scientific_data_opened": False,
                "raw_input_bytes_opened": False,
                "authorization_claimed": False,
                "gate0": "NOT_RUN",
                "training": "FORBIDDEN",
                "diagnostics": _stage_diagnostics(state),
                "absolute_paths_recorded": False,
                "source_identifiers_recorded": False,
            }
            _write_once(args.output, payload)
            print(json.dumps(payload, sort_keys=True))
            return 0
        if args.authorization is None or args.consumption_marker is None:
            payload = _blocked(
                "RUN_REQUIRES_AUTHORIZATION_AND_CONSUMPTION_MARKER", state
            )
            print(json.dumps(payload, sort_keys=True))
            return 2

        state.enter(TechnicalStage.AUTHORIZATION_CLAIM)
        claim = claim_one_run_authorization(
            args.authorization,
            args.consumption_marker,
            passport_sha256=passport_hash,
            freeze_id=freeze_id,
        )
        claimed = True
        state.complete(TechnicalStage.AUTHORIZATION_CLAIM)

        state.enter(TechnicalStage.INPUT_IDENTITY)
        input_identity = _verify_bound_inputs(
            archive_hashes,
            private_manifests,
            maples_archive=args.maples_archive,
            maples_additional=args.maples_additional,
            maples_variability=args.maples_variability,
            ddr_root=args.ddr_root,
        )
        state.complete(TechnicalStage.INPUT_IDENTITY)

        state.enter(TechnicalStage.MAPLES_SOURCE_MEASUREMENT)
        state.scientific_data_may_have_been_opened = True
        confirmatory = load_maples_test_measurements(
            args.maples_archive,
            args.maples_additional,
            args.maples_variability,
            fov_bounds=maples_bounds,
        )
        state.source_summaries["maples_dr_messidor1"] = _source_summary(confirmatory)
        state.complete(TechnicalStage.MAPLES_SOURCE_MEASUREMENT)

        state.enter(TechnicalStage.DDR_SOURCE_MEASUREMENT)
        replication = load_ddr_test_measurements(
            args.ddr_root,
            fov_bounds=ddr_bounds,
        )
        state.source_summaries["ddr"] = _source_summary(replication)
        state.complete(TechnicalStage.DDR_SOURCE_MEASUREMENT)

        state.enter(TechnicalStage.AGGREGATE_EXECUTION)
        result = execute_gate0_v06(confirmatory, replication, config)
        state.complete(TechnicalStage.AGGREGATE_EXECUTION)
        result.update(
            {
                "status": "FORMAL_GATE0_V07_COMPLETE",
                "executor_version": "0.7",
                "scientific_protocol_inherited_from": "0.6",
                "freeze_id": freeze_id,
                "freeze_passport_sha256": passport_hash,
                "authorization_sha256": claim.authorization_sha256,
                "authorization_claim_marker_sha256": claim.marker_sha256,
                "input_identity": input_identity,
                "held_out_test_opened_after_authorization_claim": True,
                "diagnostics": _stage_diagnostics(state),
                "raw_data_persisted": False,
                "raw_masks_persisted": False,
                "raw_images_persisted": False,
                "model_weights_persisted": False,
                "material_passport": {
                    "observation": [
                        "One-run authorization was atomically claimed before scientific access.",
                        "Both source endpoints were computed separately with "
                        "aggregate-only output.",
                        "Privacy-safe technical stages completed through aggregate execution.",
                    ],
                    "source_supported_semantics": [
                        "Four-axis expert lesion annotations were used under "
                        "their frozen source roles."
                    ],
                    "inference": [
                        "The decision is a preregistered expert-annotation-derived "
                        "phenomenon audit, not clinical treatment priority."
                    ],
                    "unresolved": [
                        "Patient/eye independence remains unavailable for both primary sources."
                    ],
                },
            }
        )
        state.enter(TechnicalStage.OUTPUT_PERSISTENCE)
        _write_once(args.output, result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "decision": result["decision"],
                    "freeze_id": freeze_id,
                    "training": "FORBIDDEN",
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        authorization_consumed = claimed or (
            state.current is TechnicalStage.AUTHORIZATION_CLAIM
            and args.consumption_marker is not None
            and args.consumption_marker.exists()
        )
        if not authorization_consumed:
            family = _safe_error_family(error)
            payload = _blocked(
                f"TECHNICAL_ERROR:{state.current.value}:{family}", state
            )
            print(json.dumps(payload, sort_keys=True))
            return 2
        payload = _technical_abort(
            error,
            state,
            freeze_id=freeze_id,
            passport_hash=passport_hash,
        )
        with suppress(FileExistsError, OSError):
            _write_once(args.output, payload)
        print(json.dumps(payload, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Fail-closed Gate 0 v0.8 entry point with MAPLES substage diagnostics.

Version 0.8 is a technical-diagnostic amendment only.  It inherits every
scientific parameter and decision rule from v0.6 and requires a new exact-hash
authorization before any formal scientific access.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from nts_med.cli.formal_gate0_v07 import (
    DiagnosticState,
    TechnicalStage,
    _bounds,
    _git_freeze_valid,
    _interpreter_valid,
    _mapping,
    _relative_repository_file,
    _safe_error_family,
    _source_summary,
    _stage_diagnostics,
    _verify_bound_inputs,
    _write_once,
    _yaml_mapping,
)
from nts_med.cli.formal_gate0_v07 import build_parser as _build_v07_parser
from nts_med.gate0.execution import execute_gate0_v06
from nts_med.gate0.source_io import load_ddr_test_measurements
from nts_med.gate0.source_io_v08 import (
    MaplesSourceMeasurementError,
    MaplesSubstage,
    load_maples_test_measurements_v08,
)
from nts_med.reproducibility.authorization import claim_one_run_authorization
from nts_med.reproducibility.freeze import (
    manifest_arguments,
    sha256_file,
    validate_freeze_passport,
)


def build_parser() -> argparse.ArgumentParser:
    parser = _build_v07_parser()
    parser.description = __doc__
    return parser


def _load_v08_config(root: Path, overlay_path: Path) -> dict[str, Any]:
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
        "diagnostic_base_config",
        "diagnostic_base_config_sha256",
        "diagnostics",
        "implementation",
    }
    if set(overlay) - allowed:
        raise ValueError("v0.8 overlay contains non-diagnostic fields")
    if overlay.get("schema_version") != 1 or overlay.get("name") != "gate0_dr_v0_8":
        raise ValueError("v0.8 overlay identity is invalid")
    if overlay.get("preregistration") != "metadata/preregistration/gate0_dr_v0.8.yaml":
        raise ValueError("v0.8 preregistration binding is invalid")
    if overlay.get("gate0_permitted") is not False:
        raise ValueError("v0.8 Gate 0 must remain unauthorized")
    if overlay.get("training_permitted") is not False:
        raise ValueError("v0.8 training must remain forbidden")
    if overlay.get("reuse_prior_passport_or_authorization") is not False:
        raise ValueError("prior passports and authorizations must not be reusable")

    base_relative = overlay.get("scientific_base_config")
    if base_relative != "configs/experiment/gate0_dr_v0.6.yaml":
        raise ValueError("v0.8 scientific base must be the frozen v0.6 config")
    base_path = _relative_repository_file(
        root, base_relative, name="scientific_base_config"
    )
    expected_hash = overlay.get("scientific_base_config_sha256")
    if not isinstance(expected_hash, str) or sha256_file(base_path) != expected_hash:
        raise ValueError("v0.8 scientific base config hash mismatch")

    diagnostic_relative = overlay.get("diagnostic_base_config")
    if diagnostic_relative != "configs/experiment/gate0_dr_v0.7.yaml":
        raise ValueError("v0.8 diagnostic base must be v0.7")
    diagnostic_path = _relative_repository_file(
        root, diagnostic_relative, name="diagnostic_base_config"
    )
    diagnostic_hash = overlay.get("diagnostic_base_config_sha256")
    if not isinstance(diagnostic_hash, str) or sha256_file(
        diagnostic_path
    ) != diagnostic_hash:
        raise ValueError("v0.8 diagnostic base config hash mismatch")

    diagnostics = overlay.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("v0.8 diagnostics configuration is unavailable")
    required_diagnostics = {
        "privacy_safe_stage_codes": True,
        "stable_reason_codes": True,
        "aggregate_source_counts_only": True,
        "exception_messages_in_output": False,
        "tracebacks_in_output": False,
        "scientific_parameter_changes": False,
        "maples_substages": [stage.value for stage in MaplesSubstage],
        "maples_reason_code_format": (
            "TECHNICAL_ERROR:MAPLES_SOURCE_MEASUREMENT:SUBSTAGE:ERROR_FAMILY"
        ),
    }
    if dict(diagnostics) != required_diagnostics:
        raise ValueError("v0.8 diagnostics configuration is invalid")

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
            "implementation": dict(
                _mapping(overlay.get("implementation"), name="implementation")
            ),
        }
    )
    return merged


def _v08_passport_contract(
    root: Path,
    passport_path: Path,
    config_path: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    passport = _yaml_mapping(passport_path)
    runner = passport.get("runner")
    if not isinstance(runner, Mapping) or runner.get("entry_point") != (
        "nts-med-formal-gate0-v08"
    ):
        raise ValueError("v0.8 formal runner is not frozen")
    artifacts = passport.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("freeze artifacts are unavailable")
    raw_config = artifacts.get("experiment_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("frozen v0.8 experiment config is unavailable")
    relative = raw_config.get("path")
    if relative != "configs/experiment/gate0_dr_v0.8.yaml":
        raise ValueError("frozen v0.8 experiment config path is invalid")
    frozen_config = _relative_repository_file(root, relative, name="experiment_config")
    if config_path.resolve(strict=False) != frozen_config:
        raise ValueError("the supplied config is not the frozen v0.8 config")

    config = _load_v08_config(root, config_path)
    raw_identity = config.get("input_identity")
    if not isinstance(raw_identity, Mapping):
        raise ValueError("frozen input identity configuration is unavailable")
    declared_archives = raw_identity.get("maples_archives")
    if not isinstance(declared_archives, Mapping):
        raise ValueError("frozen archive identities are unavailable")
    raw_input_files = passport.get("input_files")
    expected_names = {
        "maples_main_archive",
        "maples_additional_archive",
        "maples_variability_archive",
    }
    if not isinstance(raw_input_files, Mapping) or set(raw_input_files) != expected_names:
        raise ValueError("v0.8 archive identities are incomplete")
    if not all(isinstance(value, str) for value in raw_input_files.values()):
        raise ValueError("v0.8 archive identities are invalid")
    if dict(raw_input_files) != dict(declared_archives):
        raise ValueError("passport archives do not match the frozen scientific base")
    raw_manifests = passport.get("input_manifests")
    ddr_digest = raw_identity.get("ddr_private_manifest_sha256")
    if raw_manifests != {"ddr": ddr_digest}:
        raise ValueError("passport DDR manifest does not match the scientific base")
    return (
        {str(name): str(value) for name, value in raw_input_files.items()},
        config,
    )


def _blocked(reason: str, state: DiagnosticState) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "BLOCKED_PRE_GATE0_V08",
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
    completed_maples_diagnostics: Mapping[str, object] | None,
) -> dict[str, object]:
    diagnostics = _stage_diagnostics(state)
    failed_substage: str | None = None
    if isinstance(error, MaplesSourceMeasurementError):
        family = error.error_family
        failed_substage = error.substage.value
        diagnostics["maples_source_measurement"] = dict(error.diagnostics)
        reason = (
            f"TECHNICAL_ERROR:{state.current.value}:"
            f"{error.substage.value}:{family}"
        )
    else:
        family = _safe_error_family(error)
        if completed_maples_diagnostics is not None:
            diagnostics["maples_source_measurement"] = dict(
                completed_maples_diagnostics
            )
        reason = f"TECHNICAL_ERROR:{state.current.value}:{family}"
    return {
        "schema_version": 1,
        "status": "FORMAL_GATE0_V08_ABORTED_TECHNICAL",
        "decision": "ABORTED_TECHNICAL",
        "scientific_result": False,
        "reason_codes": [reason],
        "failed_stage": state.current.value,
        "failed_substage": failed_substage,
        "error_family": family,
        "freeze_id": freeze_id,
        "freeze_passport_sha256": passport_hash,
        "authorization_consumed": True,
        "scientific_data_may_have_been_opened": (
            state.scientific_data_may_have_been_opened
        ),
        "diagnostics": diagnostics,
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
    maples_diagnostics: Mapping[str, object] | None = None
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
            print(json.dumps(_blocked("FREEZE_PASSPORT_VALIDATION_FAILED", state), sort_keys=True))
            return 2
        freeze_id = validation.freeze_id
        if not freeze_id.startswith("gate0_dr_v0.8_"):
            print(json.dumps(_blocked("FREEZE_ID_NOT_V0_8", state), sort_keys=True))
            return 2
        if not _git_freeze_valid(root, args.passport):
            print(json.dumps(_blocked("GIT_FREEZE_CHECK_FAILED", state), sort_keys=True))
            return 2
        if not _interpreter_valid(root):
            print(json.dumps(_blocked("ISOLATED_INTERPRETER_REQUIRED", state), sort_keys=True))
            return 2
        state.complete(TechnicalStage.FREEZE_VALIDATION)

        state.enter(TechnicalStage.FROZEN_CONTRACT)
        archive_hashes, config = _v08_passport_contract(
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
                print(
                    json.dumps(
                        _blocked("REQUIRED_SOURCE_INPUT_UNAVAILABLE", state),
                        sort_keys=True,
                    )
                )
                return 2
        if not args.ddr_root.is_dir():
            print(json.dumps(_blocked("REQUIRED_SOURCE_INPUT_UNAVAILABLE", state), sort_keys=True))
            return 2
        state.complete(TechnicalStage.SOURCE_PRESENCE)

        passport_hash = sha256_file(args.passport)
        if args.mode == "preflight":
            payload = {
                "schema_version": 1,
                "status": "V0_8_PREFLIGHT_PASS_AWAITING_EXACT_HASH_AUTHORIZATION",
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
            print(
                json.dumps(
                    _blocked("RUN_REQUIRES_AUTHORIZATION_AND_CONSUMPTION_MARKER", state),
                    sort_keys=True,
                )
            )
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
        maples_outcome = load_maples_test_measurements_v08(
            args.maples_archive,
            args.maples_additional,
            args.maples_variability,
            fov_bounds=maples_bounds,
        )
        confirmatory = maples_outcome.measurements
        maples_diagnostics = maples_outcome.diagnostics
        state.source_summaries["maples_dr_messidor1"] = _source_summary(confirmatory)
        state.complete(TechnicalStage.MAPLES_SOURCE_MEASUREMENT)

        state.enter(TechnicalStage.DDR_SOURCE_MEASUREMENT)
        replication = load_ddr_test_measurements(args.ddr_root, fov_bounds=ddr_bounds)
        state.source_summaries["ddr"] = _source_summary(replication)
        state.complete(TechnicalStage.DDR_SOURCE_MEASUREMENT)

        state.enter(TechnicalStage.AGGREGATE_EXECUTION)
        result = execute_gate0_v06(confirmatory, replication, config)
        state.complete(TechnicalStage.AGGREGATE_EXECUTION)
        diagnostics = _stage_diagnostics(state)
        diagnostics["maples_source_measurement"] = dict(maples_diagnostics)
        result.update(
            {
                "status": "FORMAL_GATE0_V08_COMPLETE",
                "executor_version": "0.8",
                "scientific_protocol_inherited_from": "0.6",
                "freeze_id": freeze_id,
                "freeze_passport_sha256": passport_hash,
                "authorization_sha256": claim.authorization_sha256,
                "authorization_claim_marker_sha256": claim.marker_sha256,
                "input_identity": input_identity,
                "held_out_test_opened_after_authorization_claim": True,
                "diagnostics": diagnostics,
                "raw_data_persisted": False,
                "raw_masks_persisted": False,
                "raw_images_persisted": False,
                "model_weights_persisted": False,
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
            completed_maples_diagnostics=maples_diagnostics,
        )
        with suppress(FileExistsError, OSError):
            _write_once(args.output, payload)
        print(json.dumps(payload, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

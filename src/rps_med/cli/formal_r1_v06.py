"""Fail-closed formal RPS-Med R1 v0.6 with DDR substage diagnostics.

Version 0.6 is a technical-diagnostic amendment only.  It inherits the complete
v0.5 scientific contract and requires a new exact-hash one-run authorization.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from nts_med.reproducibility.authorization import claim_one_run_authorization
from nts_med.reproducibility.freeze import (
    manifest_arguments,
    sha256_file,
    validate_freeze_passport,
)
from rps_med.cli.formal_r1 import (
    R1DiagnosticState,
    R1Stage,
    _diagnostics,
    _git_freeze_valid,
    _interpreter_valid,
    _mapping,
    _relative_file,
    _safe_error_family,
    _verify_identity,
    _write_once,
    _yaml_mapping,
)
from rps_med.cli.formal_r1 import build_parser as _build_v05_parser
from rps_med.formal import execute_formal_r1
from rps_med.source_io import load_maples_confirmation_measurements
from rps_med.source_io_v06 import (
    DDRExclusionReason,
    DDRSourceMeasurementError,
    DDRSubstage,
    load_ddr_confirmation_measurements_v06,
)


def build_parser() -> argparse.ArgumentParser:
    parser = _build_v05_parser()
    parser.description = __doc__
    return parser


def _load_v06_config(root: Path, overlay_path: Path) -> dict[str, Any]:
    overlay = _yaml_mapping(overlay_path)
    allowed = {
        "schema_version",
        "config_id",
        "preregistration",
        "status",
        "scientific_base_config",
        "scientific_base_config_sha256",
        "scientific_parameter_changes",
        "prior_formal_result",
        "prior_formal_result_sha256",
        "diagnostics",
        "execution",
    }
    if set(overlay) != allowed:
        raise ValueError("R1 v0.6 overlay fields are invalid")
    if overlay.get("schema_version") != 1 or overlay.get("config_id") != "rps_med_r1_v0.6":
        raise ValueError("R1 v0.6 overlay identity is invalid")
    if overlay.get("preregistration") != "metadata/preregistration/rps_med_r1_v0.6.yaml":
        raise ValueError("R1 v0.6 preregistration binding is invalid")
    if overlay.get("scientific_parameter_changes") is not False:
        raise ValueError("R1 v0.6 cannot change scientific parameters")

    base_relative = overlay.get("scientific_base_config")
    if base_relative != "configs/experiment/rps_med_r1_v0.5.yaml":
        raise ValueError("R1 v0.6 scientific base must be v0.5")
    base_path = _relative_file(root, base_relative, name="scientific_base_config")
    if sha256_file(base_path) != overlay.get("scientific_base_config_sha256"):
        raise ValueError("R1 v0.6 scientific base hash mismatch")

    result_relative = overlay.get("prior_formal_result")
    if result_relative != "metadata/gates/rps_med_r1_v05_formal_result_2026-08-25_v1.yaml":
        raise ValueError("R1 v0.6 prior-result binding is invalid")
    result_path = _relative_file(root, result_relative, name="prior_formal_result")
    if sha256_file(result_path) != overlay.get("prior_formal_result_sha256"):
        raise ValueError("R1 v0.6 prior-result hash mismatch")

    diagnostics = _mapping(overlay.get("diagnostics"), name="diagnostics")
    required_diagnostics = {
        "ddr_substages": [stage.value for stage in DDRSubstage],
        "exclusion_reason_codes": [reason.value for reason in DDRExclusionReason],
        "reason_code_format": (
            "TECHNICAL_ERROR:DDR_CONFIRMATION_MEASUREMENT:SUBSTAGE:ERROR_FAMILY"
        ),
        "aggregate_reason_counts_only": True,
        "exception_messages_in_output": False,
        "tracebacks_in_output": False,
        "paths_in_output": False,
        "instance_identifiers_in_output": False,
        "row_level_measurements_in_output": False,
    }
    if dict(diagnostics) != required_diagnostics:
        raise ValueError("R1 v0.6 diagnostics contract is invalid")

    execution = _mapping(overlay.get("execution"), name="execution")
    required_execution = {
        "formal_r1": "NOT_AUTHORIZED",
        "model_training": "FORBIDDEN",
        "gate0_rerun": "FORBIDDEN",
        "prior_authorization_reuse": "FORBIDDEN",
        "output_overwrite": "FORBIDDEN",
        "result_feedback_tuning": "FORBIDDEN",
        "automatic_retry": "FORBIDDEN",
    }
    if dict(execution) != required_execution:
        raise ValueError("R1 v0.6 execution contract is invalid")

    base = _yaml_mapping(base_path)
    merged = dict(base)
    merged.update(
        {
            "config_id": "rps_med_r1_v0.6",
            "preregistration": overlay["preregistration"],
            "status": overlay["status"],
            "diagnostics": dict(diagnostics),
            "execution": dict(execution),
        }
    )
    return merged


def _v06_passport_contract(root: Path, passport_path: Path, config_path: Path) -> dict[str, Any]:
    passport = _yaml_mapping(passport_path)
    runner = _mapping(passport.get("runner"), name="runner")
    if runner.get("entry_point") != "rps-med-formal-r1-v06":
        raise ValueError("R1 v0.6 runner is not frozen")
    artifacts = _mapping(passport.get("artifacts"), name="artifacts")
    config_record = _mapping(artifacts.get("experiment_config"), name="experiment_config")
    frozen_config = _relative_file(root, config_record.get("path"), name="experiment_config")
    if config_path.resolve(strict=False) != frozen_config:
        raise ValueError("supplied config is not the frozen R1 v0.6 config")
    return _load_v06_config(root, config_path)


def _blocked(reason: str, state: R1DiagnosticState) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "BLOCKED_PRE_FORMAL_R1_V0_6",
        "reason_codes": [reason],
        "failed_stage": state.current.value,
        "diagnostics": _diagnostics(state),
        "formal_r1": "NOT_RUN",
        "scientific_data_opened": False,
        "gate0": "NOT_RERUN",
        "training": "FORBIDDEN",
    }


def _ddr_summary_from_error(state: R1DiagnosticState, error: DDRSourceMeasurementError) -> None:
    diagnostics = error.diagnostics
    observed = diagnostics.get("observed_input_count")
    eligible = diagnostics.get("measurement_eligible_count")
    excluded = diagnostics.get("measurement_exclusion_count")
    if (
        isinstance(observed, int)
        and isinstance(eligible, int)
        and isinstance(excluded, int)
    ):
        state.source_summaries["ddr"] = {
            "expected": observed,
            "eligible": eligible,
            "excluded": excluded,
        }


def _technical_abort(
    error: Exception,
    state: R1DiagnosticState,
    *,
    freeze_id: str,
    passport_hash: str,
    completed_ddr_diagnostics: Mapping[str, object] | None,
) -> dict[str, object]:
    diagnostics = _diagnostics(state)
    failed_substage: str | None = None
    family = _safe_error_family(error)
    if isinstance(error, DDRSourceMeasurementError):
        failed_substage = error.substage.value
        family = error.error_family
        _ddr_summary_from_error(state, error)
        diagnostics = _diagnostics(state)
        diagnostics["ddr_source_measurement"] = dict(error.diagnostics)
    elif completed_ddr_diagnostics is not None:
        diagnostics["ddr_source_measurement"] = dict(completed_ddr_diagnostics)
    reason = f"TECHNICAL_ERROR:{state.current.value}"
    if failed_substage is not None:
        reason += f":{failed_substage}"
    reason += f":{family}"
    return {
        "schema_version": 1,
        "status": "FORMAL_RPS_MED_R1_V0_6_ABORTED_TECHNICAL",
        "decision": "ABORTED_TECHNICAL",
        "scientific_result": False,
        "reason_codes": [reason],
        "failed_stage": state.current.value,
        "failed_substage": failed_substage,
        "error_family": family,
        "freeze_id": freeze_id,
        "freeze_passport_sha256": passport_hash,
        "authorization_consumed": True,
        "diagnostics": diagnostics,
        "formal_r1": "NOT_ADJUDICATED",
        "gate0": "NOT_RERUN",
        "automatic_retry": "FORBIDDEN",
        "training": "FORBIDDEN",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repository_root.resolve()
    state = R1DiagnosticState()
    claimed = False
    passport_hash = ""
    freeze_id = ""
    completed_ddr_diagnostics: Mapping[str, object] | None = None
    try:
        state.enter(R1Stage.STARTUP_GUARD)
        if args.output.exists() or (
            args.consumption_marker is not None and args.consumption_marker.exists()
        ):
            raise FileExistsError("formal R1 v0.6 output or marker already exists")
        private_manifests = manifest_arguments(args.private_manifest)
        if set(private_manifests) != {"ddr"}:
            raise ValueError("formal R1 v0.6 requires exactly the DDR private manifest")
        state.complete(R1Stage.STARTUP_GUARD)

        state.enter(R1Stage.FREEZE_VALIDATION)
        validation = validate_freeze_passport(
            root, args.passport, private_manifests=private_manifests
        )
        if not validation.valid or validation.freeze_id is None:
            print(json.dumps(_blocked("FREEZE_PASSPORT_VALIDATION_FAILED", state)))
            return 2
        freeze_id = validation.freeze_id
        if not freeze_id.startswith("rps_med_r1_v0.6_"):
            print(json.dumps(_blocked("FREEZE_ID_NOT_RPS_R1_V0_6", state)))
            return 2
        if not _git_freeze_valid(root, args.passport):
            print(json.dumps(_blocked("GIT_FREEZE_CHECK_FAILED", state)))
            return 2
        if not _interpreter_valid(root):
            print(json.dumps(_blocked("ISOLATED_INTERPRETER_REQUIRED", state)))
            return 2
        state.complete(R1Stage.FREEZE_VALIDATION)

        state.enter(R1Stage.FROZEN_CONTRACT)
        config = _v06_passport_contract(root, args.passport, args.config)
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
                "status": "RPS_MED_R1_V0_6_PREFLIGHT_PASS_AWAITING_EXACT_HASH_AUTHORIZATION",
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

        boundary = _mapping(config.get("confirmation_boundary"), name="confirmation_boundary")
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
        ddr_outcome = load_ddr_confirmation_measurements_v06(args.ddr_root)
        ddr = ddr_outcome.measurements
        completed_ddr_diagnostics = ddr_outcome.diagnostics
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
        diagnostics = _diagnostics(state)
        diagnostics["ddr_source_measurement"] = dict(completed_ddr_diagnostics)
        result.update(
            {
                "status": "FORMAL_RPS_MED_R1_V0_6_COMPLETE",
                "freeze_id": freeze_id,
                "freeze_passport_sha256": passport_hash,
                "authorization_sha256": claim.authorization_sha256,
                "authorization_marker_sha256": claim.marker_sha256,
                "authorization_consumed": True,
                "input_identity": input_identity,
                "confirmation_opened_after_authorization_claim": True,
                "diagnostics": diagnostics,
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
        payload = _technical_abort(
            error,
            state,
            freeze_id=freeze_id,
            passport_hash=passport_hash,
            completed_ddr_diagnostics=completed_ddr_diagnostics,
        )
        with suppress(FileExistsError, OSError, ValueError):
            _write_once(args.output, payload)
        print(json.dumps({"status": payload["status"], "failed_stage": state.current.value}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

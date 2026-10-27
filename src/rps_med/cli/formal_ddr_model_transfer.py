"""Consume one authorization and run frozen TJDR-to-DDR model transfer once."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from nts_med.reproducibility.authorization import claim_one_run_authorization
from nts_med.reproducibility.freeze import (
    load_yaml_mapping,
    sha256_file,
    validate_freeze_passport,
)
from rps_med.cli.formal_r1 import _git_freeze_valid, _interpreter_valid, _write_once
from rps_med.ddr_model_transfer import evaluate_ddr_model_transfer
from rps_med.source_io import validate_ddr_confirmation_manifest

_SCOPE = "RPS_MED_DDR_MODEL_TRANSFER_ONCE"
_SEEDS = (20260826, 20260827, 20260828)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--passport", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--consumption-marker", type=Path)
    parser.add_argument("--ddr-manifest", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--development-config", type=Path, required=True)
    parser.add_argument("--weights-directory", type=Path, required=True)
    parser.add_argument("--ddr-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} is invalid")
    return value


def _input_file_checks(
    passport: Mapping[str, Any],
    cache: Path,
    weights: Path,
    config: Path,
    development_config: Path,
) -> dict[str, bool]:
    expected = _mapping(passport.get("input_files"), "input_files")
    artifacts = _mapping(passport.get("artifacts"), "artifacts")
    transfer_record = _mapping(artifacts.get("transfer_config"), "transfer_config")
    development_record = _mapping(
        artifacts.get("development_config"), "development_config"
    )
    checks = {
        "development_cache": sha256_file(cache) == expected.get("development_cache"),
        "transfer_config": sha256_file(config) == transfer_record.get("sha256"),
        "development_config": sha256_file(development_config)
        == development_record.get("sha256"),
    }
    for seed in _SEEDS:
        for role, prefix in (("burden", "burden_encoder"), ("pair", "rps_multitask_pair")):
            path = weights / f"{prefix}_seed{seed}.pt"
            checks[f"{role}_seed_{seed}"] = path.is_file() and sha256_file(path) == expected.get(
                f"{role}_seed_{seed}"
            )
    return dict(sorted(checks.items()))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    claimed = False
    freeze_id = ""
    passport_hash = ""
    stage = "STARTUP_GUARD"
    try:
        if args.output.exists() or (
            args.consumption_marker is not None and args.consumption_marker.exists()
        ):
            raise FileExistsError("DDR transfer output or marker already exists")
        stage = "FREEZE_VALIDATION"
        validation = validate_freeze_passport(
            args.repository_root,
            args.passport,
            private_manifests={"ddr": args.ddr_manifest},
        )
        if not validation.valid or validation.freeze_id is None:
            raise PermissionError("DDR transfer freeze validation failed")
        freeze_id = validation.freeze_id
        if not freeze_id.startswith("rps_med_ddr_model_transfer_v0.1_"):
            raise PermissionError("DDR transfer freeze identity is invalid")
        if not _git_freeze_valid(args.repository_root, args.passport):
            raise PermissionError("DDR transfer Git freeze validation failed")
        if not _interpreter_valid(args.repository_root):
            raise PermissionError("isolated interpreter is required")
        passport_hash = sha256_file(args.passport)
        passport = load_yaml_mapping(args.passport)
        identity = _input_file_checks(
            passport,
            args.train_cache,
            args.weights_directory,
            args.config,
            args.development_config,
        )
        if not all(identity.values()) or not args.ddr_root.is_dir():
            raise PermissionError("DDR transfer input identity failed")
        if args.mode == "preflight":
            print(
                json.dumps(
                    {
                        "status": "RPS_MED_DDR_MODEL_TRANSFER_PREFLIGHT_PASS",
                        "freeze_id": freeze_id,
                        "freeze_passport_sha256": passport_hash,
                        "input_identity": identity,
                        "scientific_data_opened": False,
                        "authorization_consumed": False,
                    }
                )
            )
            return 0
        if args.authorization is None or args.consumption_marker is None:
            raise PermissionError("DDR transfer run requires authorization and marker")
        stage = "AUTHORIZATION_CLAIM"
        claim = claim_one_run_authorization(
            args.authorization,
            args.consumption_marker,
            passport_sha256=passport_hash,
            freeze_id=freeze_id,
            authorization_scope=_SCOPE,
        )
        claimed = True
        stage = "DDR_INPUT_IDENTITY"
        manifest = validate_ddr_confirmation_manifest(args.ddr_root, args.ddr_manifest)
        if manifest.get("valid") is not True:
            raise ValueError("DDR valid bytes differ from the bound private manifest")
        stage = "MODEL_TRANSFER_EVALUATION"
        result = evaluate_ddr_model_transfer(
            args.train_cache,
            args.ddr_root,
            args.development_config,
            args.weights_directory,
            expected_count=149,
            bootstrap_replicates=5000,
            bootstrap_seed=20260831,
        )
        payload = {
            "schema_version": 1,
            **result,
            "freeze_id": freeze_id,
            "freeze_passport_sha256": passport_hash,
            "authorization_consumed": True,
            "authorization_sha256": claim.authorization_sha256,
            "authorization_marker_sha256": claim.marker_sha256,
            "input_identity": identity,
            "ddr_manifest_validation": manifest,
        }
        stage = "OUTPUT_PERSISTENCE"
        _write_once(args.output, payload)
        print(json.dumps({"status": payload["status"], "decision": payload["decision"]}))
        return 0
    except Exception:
        if claimed:
            payload = {
                "schema_version": 1,
                "status": "RPS_MED_DDR_MODEL_TRANSFER_ABORTED_TECHNICAL",
                "decision": "NOT_ADJUDICATED",
                "failed_stage": stage,
                "reason_code": f"TECHNICAL_ERROR:{stage}",
                "freeze_id": freeze_id,
                "freeze_passport_sha256": passport_hash,
                "authorization_consumed": True,
                "exception_message_recorded": False,
                "source_identifiers_recorded": False,
                "automatic_retry": False,
                "model_retraining": False,
            }
            with suppress(FileExistsError, OSError, ValueError):
                _write_once(args.output, payload)
            print(json.dumps({"status": payload["status"], "failed_stage": stage}))
            return 2
        print(json.dumps({"status": "BLOCKED_PRE_DDR_MODEL_TRANSFER", "failed_stage": stage}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

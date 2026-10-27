"""Consume one authorization and evaluate the frozen TJDR RPS-Med model once."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nts_med.reproducibility.freeze import (
    load_yaml_mapping,
    sha256_file,
    validate_authorization,
    validate_freeze_passport,
)
from rps_med.cli.formal_r1 import _write_once
from rps_med.model_evaluation import evaluate_formal_tjdr_model_test

_SEEDS = (20260826, 20260827, 20260828)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--passport", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--consumption-marker", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--test-archive", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _expected_inputs(passport: Mapping[str, Any]) -> Mapping[str, str]:
    values = passport.get("input_files")
    if not isinstance(values, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in values.items()
    ):
        raise ValueError("formal model passport input hashes are invalid")
    return {str(key): str(value) for key, value in values.items()}


def _input_identity(
    passport: Mapping[str, Any],
    *,
    train_cache: Path,
    test_archives: Sequence[Path],
    weights_directory: Path,
) -> dict[str, bool]:
    expected = _expected_inputs(passport)
    checks = {
        "development_cache": sha256_file(train_cache) == expected.get("development_cache"),
        "test_archive_count": len(test_archives) == 1,
        "test_archive": len(test_archives) == 1
        and sha256_file(test_archives[0]) == expected.get("tjdr_test"),
    }
    for seed in _SEEDS:
        for role, prefix in (
            ("burden", "burden_encoder"),
            ("pair", "rps_multitask_pair"),
        ):
            key = f"{role}_seed_{seed}"
            path = weights_directory / f"{prefix}_seed{seed}.pt"
            checks[key] = path.is_file() and sha256_file(path) == expected.get(key)
    return dict(sorted(checks.items()))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists() or args.consumption_marker.exists():
        raise FileExistsError("formal TJDR model output or marker already exists")
    passport_hash = sha256_file(args.passport)
    passport = load_yaml_mapping(args.passport)
    freeze_id = passport.get("freeze_id")
    if not isinstance(freeze_id, str):
        raise ValueError("formal model freeze id is invalid")
    freeze_validation = validate_freeze_passport(
        args.repository_root, args.passport, private_manifests={}
    )
    authorization_valid, authorization_reasons = validate_authorization(
        args.authorization,
        passport_sha256=passport_hash,
        freeze_id=freeze_id,
        authorization_scope="RPS_MED_TJDR_MODEL_TEST_ONCE",
    )
    identity = _input_identity(
        passport,
        train_cache=args.train_cache,
        test_archives=args.test_archive,
        weights_directory=args.weights_directory,
    )
    if not freeze_validation.valid or not authorization_valid or not all(identity.values()):
        raise PermissionError("formal TJDR model preflight failed")
    marker = {
        "status": "FORMAL_TJDR_MODEL_TEST_AUTHORIZATION_CONSUMED",
        "freeze_id": freeze_id,
        "freeze_passport_sha256": passport_hash,
        "authorization_sha256": sha256_file(args.authorization),
        "execution_count": 1,
        "input_identity": identity,
        "authorization_reason_codes": list(authorization_reasons),
        "official_test_scientific_access_started": True,
        "automatic_retry": False,
    }
    _write_once(args.consumption_marker, marker)
    try:
        result = evaluate_formal_tjdr_model_test(
            args.train_cache,
            args.test_archive,
            args.config,
            args.weights_directory,
            bootstrap_replicates=5000,
            bootstrap_seed=20260829,
        )
        payload = {
            "schema_version": 1,
            **result,
            "freeze_id": freeze_id,
            "freeze_passport_sha256": passport_hash,
            "authorization_consumed": True,
            "input_identity": identity,
        }
    except (OSError, RuntimeError, TypeError, ValueError):
        payload = {
            "schema_version": 1,
            "status": "RPS_MED_TJDR_MODEL_FORMAL_TEST_ABORTED_TECHNICAL",
            "decision": "NOT_ADJUDICATED",
            "reason_code": "FORMAL_MODEL_EVALUATION_SAFE_ERROR",
            "freeze_id": freeze_id,
            "freeze_passport_sha256": passport_hash,
            "authorization_consumed": True,
            "input_identity": identity,
            "exception_message_recorded": False,
            "automatic_retry": False,
        }
    _write_once(args.output, payload)
    print(json.dumps({"status": payload["status"], "decision": payload["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

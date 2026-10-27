"""Single fail-closed formal Gate 0 v0.6 scientific entry point."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from nts_med.gate0.execution import execute_gate0_v06
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


def _v06_passport_contract(
    root: Path,
    passport_path: Path,
    config_path: Path,
) -> dict[str, str]:
    passport = _yaml_mapping(passport_path)
    runner = passport.get("runner")
    if not isinstance(runner, Mapping) or runner.get("entry_point") != (
        "nts-med-formal-gate0-v06"
    ):
        raise ValueError("v0.6 formal runner is not frozen")
    artifacts = passport.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("freeze artifacts are unavailable")
    raw_config = artifacts.get("experiment_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("frozen experiment config is unavailable")
    relative = raw_config.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("frozen experiment config path is invalid")
    portable = PurePosixPath(relative)
    if portable.is_absolute() or ".." in portable.parts:
        raise ValueError("frozen experiment config path is invalid")
    frozen_config = (root / Path(*portable.parts)).resolve(strict=False)
    if config_path.resolve(strict=False) != frozen_config:
        raise ValueError("the supplied config is not the frozen experiment config")
    config = _yaml_mapping(config_path)
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
        raise ValueError("v0.6 archive identities are incomplete")
    if not all(isinstance(value, str) for value in raw_input_files.values()):
        raise ValueError("v0.6 archive identities are invalid")
    if dict(raw_input_files) != dict(declared_archives):
        raise ValueError("passport archives do not match the frozen config")
    raw_manifests = passport.get("input_manifests")
    ddr_digest = raw_identity.get("ddr_private_manifest_sha256")
    if raw_manifests != {"ddr": ddr_digest}:
        raise ValueError("passport DDR manifest does not match the frozen config")
    return {str(name): str(value) for name, value in raw_input_files.items()}


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


def _write_once(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError("formal output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _blocked(reason: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "BLOCKED_PRE_GATE0_V06",
        "reason_codes": [reason],
        "gate0": "NOT_RUN",
        "scientific_data_opened": False,
        "training": "FORBIDDEN",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repository_root.resolve()
    claimed = False
    passport_hash = ""
    freeze_id = ""
    try:
        if args.output.exists() or (
            args.consumption_marker is not None and args.consumption_marker.exists()
        ):
            raise FileExistsError("formal output or consumption marker already exists")
        private_manifests = manifest_arguments(args.private_manifest)
        validation = validate_freeze_passport(
            root, args.passport, private_manifests=private_manifests
        )
        if not validation.valid or validation.freeze_id is None:
            payload = _blocked("FREEZE_PASSPORT_VALIDATION_FAILED")
            print(json.dumps(payload, sort_keys=True))
            return 2
        freeze_id = validation.freeze_id
        if not freeze_id.startswith("gate0_dr_v0.6_"):
            payload = _blocked("FREEZE_ID_NOT_V0_6")
            print(json.dumps(payload, sort_keys=True))
            return 2
        if not _git_freeze_valid(root, args.passport):
            payload = _blocked("GIT_FREEZE_CHECK_FAILED")
            print(json.dumps(payload, sort_keys=True))
            return 2
        if not _interpreter_valid(root):
            payload = _blocked("ISOLATED_INTERPRETER_REQUIRED")
            print(json.dumps(payload, sort_keys=True))
            return 2
        archive_hashes = _v06_passport_contract(root, args.passport, args.config)
        config = _yaml_mapping(args.config)
        if config.get("preregistration") != "metadata/preregistration/gate0_dr_v0.6.yaml":
            payload = _blocked("V0_6_CONFIG_REQUIRED")
            print(json.dumps(payload, sort_keys=True))
            return 2
        for path in (
            args.maples_archive,
            args.maples_additional,
            args.maples_variability,
        ):
            if not path.is_file():
                payload = _blocked("REQUIRED_SOURCE_INPUT_UNAVAILABLE")
                print(json.dumps(payload, sort_keys=True))
                return 2
        if not args.ddr_root.is_dir():
            payload = _blocked("REQUIRED_SOURCE_INPUT_UNAVAILABLE")
            print(json.dumps(payload, sort_keys=True))
            return 2

        passport_hash = sha256_file(args.passport)
        if args.mode == "preflight":
            payload = {
                "schema_version": 1,
                "status": "V0_6_PREFLIGHT_PASS_AWAITING_EXACT_HASH_AUTHORIZATION",
                "freeze_id": freeze_id,
                "freeze_passport_sha256": passport_hash,
                "scientific_data_opened": False,
                "raw_input_bytes_opened": False,
                "authorization_claimed": False,
                "gate0": "NOT_RUN",
                "training": "FORBIDDEN",
                "absolute_paths_recorded": False,
                "source_identifiers_recorded": False,
            }
            _write_once(args.output, payload)
            print(json.dumps(payload, sort_keys=True))
            return 0
        if args.authorization is None or args.consumption_marker is None:
            payload = _blocked("RUN_REQUIRES_AUTHORIZATION_AND_CONSUMPTION_MARKER")
            print(json.dumps(payload, sort_keys=True))
            return 2

        claim = claim_one_run_authorization(
            args.authorization,
            args.consumption_marker,
            passport_sha256=passport_hash,
            freeze_id=freeze_id,
        )
        claimed = True

        input_identity = _verify_bound_inputs(
            archive_hashes,
            private_manifests,
            maples_archive=args.maples_archive,
            maples_additional=args.maples_additional,
            maples_variability=args.maples_variability,
            ddr_root=args.ddr_root,
        )

        confirmatory = load_maples_test_measurements(
            args.maples_archive,
            args.maples_additional,
            args.maples_variability,
            fov_bounds=_bounds(config, "maples_dr_messidor1"),
        )
        replication = load_ddr_test_measurements(
            args.ddr_root,
            fov_bounds=_bounds(config, "ddr"),
        )
        result = execute_gate0_v06(confirmatory, replication, config)
        result.update(
            {
                "freeze_id": freeze_id,
                "freeze_passport_sha256": passport_hash,
                "authorization_sha256": claim.authorization_sha256,
                "authorization_claim_marker_sha256": claim.marker_sha256,
                "input_identity": input_identity,
                "held_out_test_opened_after_authorization_claim": True,
                "raw_data_persisted": False,
                "raw_masks_persisted": False,
                "raw_images_persisted": False,
                "model_weights_persisted": False,
                "material_passport": {
                    "observation": [
                        "One-run authorization was atomically claimed before scientific access.",
                        "Both source endpoints were computed separately with "
                        "aggregate-only output.",
                    ],
                    "source_supported_semantics": [
                        "Four-axis expert lesion annotations were used under their "
                        "frozen source roles."
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
    except (OSError, PermissionError, TypeError, ValueError, yaml.YAMLError) as error:
        if not claimed:
            payload = _blocked(f"PREFLIGHT_ERROR:{type(error).__name__}")
            print(json.dumps(payload, sort_keys=True))
            return 2
        payload = {
            "schema_version": 1,
            "status": "FORMAL_GATE0_V06_ABORTED_TECHNICAL",
            "decision": "ABORTED_TECHNICAL",
            "reason_codes": [f"EXECUTOR_ERROR:{type(error).__name__}"],
            "freeze_id": freeze_id,
            "freeze_passport_sha256": passport_hash,
            "authorization_consumed": True,
            "scientific_data_may_have_been_opened": True,
            "rerun_without_new_version_and_authorization": "FORBIDDEN",
            "training": "FORBIDDEN",
        }
        with suppress(FileExistsError, OSError):
            _write_once(args.output, payload)
        print(json.dumps(payload, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate a separate, non-self-referential Gate 0 freeze passport."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class FreezeValidation:
    """Aggregate path-safe result of freeze-passport validation."""

    valid: bool
    checks: Mapping[str, bool]
    reason_codes: tuple[str, ...]
    freeze_id: str | None


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of one file without exposing its path."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return value


def _relative_path(root: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    portable = PurePosixPath(value)
    if portable.is_absolute() or ".." in portable.parts:
        raise ValueError(f"{name} must stay within the repository")
    candidate = (root / Path(*portable.parts)).resolve(strict=False)
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{name} must stay within the repository") from error
    return candidate


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load one YAML mapping with a deterministic error boundary."""

    with Path(path).open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise ValueError("YAML document must be a string-keyed mapping")
    return loaded


def validate_freeze_passport(
    repository_root: str | Path,
    passport_path: str | Path,
    *,
    private_manifests: Mapping[str, str | Path] | None = None,
) -> FreezeValidation:
    """Validate tracked artifact hashes and optional private input manifests."""

    root = Path(repository_root).resolve()
    checks: dict[str, bool] = {}
    reasons: list[str] = []
    freeze_id: str | None = None
    try:
        passport = load_yaml_mapping(passport_path)
        raw_id = passport.get("freeze_id")
        freeze_id = raw_id if isinstance(raw_id, str) and raw_id else None
        checks["schema_version"] = passport.get("schema_version") == 1
        checks["freeze_id"] = freeze_id is not None
        checks["status"] = passport.get("status") == "FROZEN_READY_AWAITING_AUTHORIZATION"
        execution = _mapping(passport.get("execution"), name="execution")
        checks["gate0_not_run"] = execution.get("gate0") == "NOT_RUN"
        checks["training_forbidden"] = execution.get("training") == "FORBIDDEN"
        checks["authorization_required"] = execution.get("authorization_required") is True

        artifacts = _mapping(passport.get("artifacts"), name="artifacts")
        checks["artifacts_present"] = bool(artifacts)
        for label, raw_record in artifacts.items():
            record = _mapping(raw_record, name=f"artifacts.{label}")
            artifact = _relative_path(root, record.get("path"), name=f"artifacts.{label}.path")
            expected = record.get("sha256")
            match = (
                isinstance(expected, str)
                and len(expected) == 64
                and artifact.is_file()
                and sha256_file(artifact) == expected
            )
            checks[f"artifact:{label}"] = match

        expected_inputs = _mapping(passport.get("input_manifests"), name="input_manifests")
        supplied = private_manifests or {}
        checks["input_manifest_names"] = set(supplied) == set(expected_inputs)
        for label, expected in expected_inputs.items():
            path = supplied.get(label)
            match = (
                isinstance(expected, str)
                and len(expected) == 64
                and path is not None
                and Path(path).is_file()
                and sha256_file(Path(path)) == expected
            )
            checks[f"input_manifest:{label}"] = match
        raw_input_files = passport.get("input_files", {})
        input_files = _mapping(raw_input_files, name="input_files")
        checks["input_file_hashes_valid"] = all(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for digest in input_files.values()
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        reasons.append("FREEZE_PASSPORT_INVALID_OR_UNREADABLE")
    for name, passed in checks.items():
        if not passed:
            reasons.append(f"CHECK_FAILED:{name}")
    return FreezeValidation(
        valid=not reasons and bool(checks),
        checks=dict(sorted(checks.items())),
        reason_codes=tuple(sorted(set(reasons))),
        freeze_id=freeze_id,
    )


def validate_authorization(
    authorization_path: str | Path,
    *,
    passport_sha256: str,
    freeze_id: str,
    authorization_scope: str = "FORMAL_GATE0_ONCE",
) -> tuple[bool, tuple[str, ...]]:
    """Validate a one-run authorization record bound to one passport."""

    reasons: list[str] = []
    try:
        record = load_yaml_mapping(authorization_path)
        if record.get("schema_version") != 1:
            reasons.append("AUTHORIZATION_SCHEMA_INVALID")
        if record.get("authorization") != authorization_scope:
            reasons.append("AUTHORIZATION_SCOPE_INVALID")
        if record.get("authorized") is not True:
            reasons.append("AUTHORIZATION_NOT_GRANTED")
        if record.get("freeze_id") != freeze_id:
            reasons.append("AUTHORIZATION_FREEZE_ID_MISMATCH")
        if record.get("freeze_passport_sha256") != passport_sha256:
            reasons.append("AUTHORIZATION_PASSPORT_HASH_MISMATCH")
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        reasons.append("AUTHORIZATION_INVALID_OR_UNREADABLE")
    return not reasons, tuple(sorted(set(reasons)))


def protected_paths(passport_path: str | Path) -> tuple[str, ...]:
    """Return normalized repository-relative paths bound by a passport."""

    passport = load_yaml_mapping(passport_path)
    artifacts = _mapping(passport.get("artifacts"), name="artifacts")
    paths: list[str] = []
    for label, raw_record in artifacts.items():
        record = _mapping(raw_record, name=f"artifacts.{label}")
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("artifact path must be a non-empty string")
        paths.append(PurePosixPath(raw_path).as_posix())
    return tuple(sorted(paths))


def manifest_arguments(values: Sequence[str]) -> dict[str, Path]:
    """Parse repeated ``NAME=PATH`` private-manifest CLI values."""

    parsed: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in parsed:
            raise ValueError("private manifests must use unique NAME=PATH values")
        parsed[name] = Path(path)
    return parsed


__all__ = [
    "FreezeValidation",
    "load_yaml_mapping",
    "manifest_arguments",
    "protected_paths",
    "sha256_file",
    "validate_authorization",
    "validate_freeze_passport",
]

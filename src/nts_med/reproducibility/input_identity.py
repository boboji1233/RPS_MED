"""Path-safe verification that a local directory still matches a private manifest."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from nts_med.data.manifests import ManifestStatus, build_manifest


@dataclass(frozen=True, slots=True)
class InputIdentityValidation:
    """Aggregate-only input identity result without source paths or identifiers."""

    valid: bool
    expected_file_count: int
    observed_file_count: int
    reason_codes: tuple[str, ...]


def _relative_path(value: str) -> str:
    if not value or value == "." or "\x00" in value:
        raise ValueError("manifest path is invalid")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError("manifest path is not relative")
    if ".." in posix.parts or ".." in windows.parts:
        raise ValueError("manifest path escapes its root")
    return posix.as_posix()


def _expected_rows(path: Path) -> dict[str, tuple[int, str]]:
    expected: dict[str, tuple[int, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"relative_path", "size_bytes", "sha256", "status"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("private manifest schema is invalid")
        for row in reader:
            relative = _relative_path(row["relative_path"])
            if relative in expected or row["status"] != ManifestStatus.OK.value:
                raise ValueError("private manifest row is invalid")
            size = int(row["size_bytes"])
            digest = row["sha256"]
            if (
                size < 0
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("private manifest identity value is invalid")
            expected[relative] = (size, digest)
    if not expected:
        raise ValueError("private manifest is empty")
    return expected


def validate_directory_manifest(
    root: str | Path,
    private_manifest: str | Path,
) -> InputIdentityValidation:
    """Hash every root-contained file and compare it with one private CSV manifest."""

    reasons: list[str] = []
    expected: Mapping[str, tuple[int, str]] = {}
    observed_count = 0
    try:
        expected = _expected_rows(Path(private_manifest))
        observed_records = build_manifest(root, include_image_metadata=False)
        observed_count = len(observed_records)
        observed = {
            record.relative_path: (record.size_bytes, record.sha256)
            for record in observed_records
            if record.status is ManifestStatus.OK
        }
        if len(observed) != len(observed_records):
            reasons.append("CURRENT_INPUT_FILE_READ_ERROR")
        if set(observed) != set(expected):
            reasons.append("INPUT_FILE_SET_MISMATCH")
        elif any(observed[relative] != identity for relative, identity in expected.items()):
            reasons.append("INPUT_FILE_IDENTITY_MISMATCH")
    except (OSError, TypeError, ValueError):
        reasons.append("INPUT_MANIFEST_OR_ROOT_INVALID")
    return InputIdentityValidation(
        valid=not reasons and bool(expected),
        expected_file_count=len(expected),
        observed_file_count=observed_count,
        reason_codes=tuple(sorted(set(reasons))),
    )


__all__ = ["InputIdentityValidation", "validate_directory_manifest"]

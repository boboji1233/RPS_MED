"""Dataset-adapter protocol and neutral discovery result types."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

DATA_NOT_AVAILABLE = "DATA_NOT_AVAILABLE"


class AdapterStatus(StrEnum):
    """Structural adapter states, not license or scientific qualification."""

    READY = "READY"
    INCOMPLETE = "INCOMPLETE"
    DATA_NOT_AVAILABLE = "DATA_NOT_AVAILABLE"
    INVALID = "INVALID"


class AdapterNotReadyError(RuntimeError):
    """Raised when samples are requested from a non-ready adapter.

    Messages intentionally contain only dataset-level status codes.  Source
    paths and identifiers can contain patient information and must not leak
    through exceptions or logs.
    """


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Result of locating a configured dataset without inventing counts."""

    status: AdapterStatus
    root: Path
    files: tuple[Path, ...] | None = None
    message: str = ""


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Schema/access validation result returned by an adapter."""

    status: AdapterStatus
    issues: tuple[str, ...] = ()
    message: str = ""


@dataclass(frozen=True, slots=True)
class DatasetSample:
    """Dataset-neutral sample skeleton; adapters retain source-specific fields."""

    sample_id: str
    image_path: Path
    dataset_name: str = ""
    official_split: str | None = None
    patient_id: str | None = None
    eye_id: str | None = None
    grade_fields: Mapping[str, str | int | float | None] = field(default_factory=dict)
    quality_fields: Mapping[str, str | int | float | None] = field(default_factory=dict)
    lesion_masks: Mapping[str, Path] = field(default_factory=dict)
    metadata: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)


@runtime_checkable
class DatasetAdapter(Protocol):
    """Minimal contract for dataset-specific discovery and sample mapping."""

    def discover(self) -> DiscoveryResult:
        """Locate configured data and return ``DATA_NOT_AVAILABLE`` if absent."""

    def validate(self) -> ValidationResult:
        """Validate only known, documented structure and metadata."""

    def samples(self) -> Iterable[DatasetSample]:
        """Iterate normalized samples after successful validation."""

    def audit_facts(self) -> Mapping[str, object]:
        """Return aggregate, path-free facts suitable for an M1 audit."""


def unavailable_discovery(root: str | Path) -> DiscoveryResult:
    """Create the canonical no-data response without a fabricated file count."""

    return DiscoveryResult(
        status=AdapterStatus.DATA_NOT_AVAILABLE,
        root=Path(root),
        files=None,
        message=DATA_NOT_AVAILABLE,
    )


_DATASET_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")


def opaque_sample_id(dataset_name: str, source_key: str) -> str:
    """Return a stable dataset-scoped identifier without exposing ``source_key``.

    The domain separator prevents accidental reuse of a digest made for a
    different purpose.  The source key is never included in errors or output.
    """

    if not isinstance(dataset_name, str) or not _DATASET_NAME.fullmatch(dataset_name):
        raise ValueError("dataset_name must be a canonical lowercase identifier")
    if not isinstance(source_key, str) or not source_key:
        raise ValueError("source_key must be a non-empty string")
    digest = hashlib.sha256(
        b"nts-med/sample-id/v1\0"
        + dataset_name.encode("utf-8")
        + b"\0"
        + source_key.encode("utf-8")
    ).hexdigest()
    return f"{dataset_name}_{digest[:24]}"


def require_ready(dataset_name: str, validation: ValidationResult) -> None:
    """Fail safely when a caller tries to enumerate incomplete samples."""

    if validation.status is not AdapterStatus.READY:
        code = dataset_name.upper().replace("-", "_")
        raise AdapterNotReadyError(f"{code}_ADAPTER_NOT_READY:{validation.status.value}")

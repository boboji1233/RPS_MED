"""Basic manifest-backed dataset audit states."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from nts_med.data.adapters.base import DATA_NOT_AVAILABLE
from nts_med.data.manifests import ManifestRecord, ManifestStatus, build_manifest


class AuditStatus(StrEnum):
    """Dataset availability and integrity states used before Gate 0."""

    READY = "READY"
    ISSUES_FOUND = "ISSUES_FOUND"
    DATA_NOT_AVAILABLE = "DATA_NOT_AVAILABLE"


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    """A non-clinical, file-level dataset audit summary."""

    status: AuditStatus
    root: str
    file_count: int | None
    total_bytes: int | None
    corrupt_files: tuple[str, ...] = ()
    unreadable_files: tuple[str, ...] = ()
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "status": self.status.value,
            "root": self.root,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "corrupt_files": list(self.corrupt_files),
            "unreadable_files": list(self.unreadable_files),
            "message": self.message,
        }


def audit_manifest(records: Iterable[ManifestRecord], *, root: str | Path = "") -> DatasetAudit:
    """Summarize manifest integrity without interpreting medical labels."""

    materialized = sorted(records, key=lambda record: record.relative_path)
    if not materialized:
        return DatasetAudit(
            status=AuditStatus.DATA_NOT_AVAILABLE,
            root=str(root),
            file_count=None,
            total_bytes=None,
            message=DATA_NOT_AVAILABLE,
        )
    corrupt = tuple(
        record.relative_path
        for record in materialized
        if record.status is ManifestStatus.CORRUPT_IMAGE
    )
    unreadable = tuple(
        record.relative_path
        for record in materialized
        if record.status is ManifestStatus.FILE_READ_ERROR
    )
    has_issues = bool(corrupt or unreadable)
    return DatasetAudit(
        status=AuditStatus.ISSUES_FOUND if has_issues else AuditStatus.READY,
        root=str(root),
        file_count=len(materialized),
        total_bytes=sum(
            record.size_bytes for record in materialized if record.size_bytes is not None
        ),
        corrupt_files=corrupt,
        unreadable_files=unreadable,
        message="FILE_INTEGRITY_ISSUES" if has_issues else "FILE_AUDIT_COMPLETE",
    )


def audit_dataset(root: str | Path, *, include_image_metadata: bool = True) -> DatasetAudit:
    """Audit an available root or return an explicit no-data state.

    Counts are ``None`` when data are absent so the status cannot be mistaken
    for a genuinely discovered empty dataset.
    """

    path = Path(root)
    if not path.exists() or not path.is_dir():
        return DatasetAudit(
            status=AuditStatus.DATA_NOT_AVAILABLE,
            root=str(path),
            file_count=None,
            total_bytes=None,
            message=DATA_NOT_AVAILABLE,
        )
    records = build_manifest(path, include_image_metadata=include_image_metadata)
    return audit_manifest(records, root=path)

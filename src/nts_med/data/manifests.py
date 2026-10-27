"""Deterministic, root-bounded dataset manifest construction."""

from __future__ import annotations

import csv
import hashlib
import importlib
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Final

DEFAULT_IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "relative_path",
    "size_bytes",
    "suffix",
    "sha256",
    "width",
    "height",
    "channels",
    "status",
    "error",
)


class ManifestStatus(StrEnum):
    """Per-file read/decode status."""

    OK = "OK"
    CORRUPT_IMAGE = "CORRUPT_IMAGE"
    FILE_READ_ERROR = "FILE_READ_ERROR"


class ManifestDependencyError(RuntimeError):
    """Raised when an explicitly requested optional output feature is absent."""


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    """Stable manifest representation for one root-contained file."""

    relative_path: str
    size_bytes: int | None
    suffix: str
    sha256: str | None
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    status: ManifestStatus = ManifestStatus.OK
    error: str | None = None

    def to_row(self) -> dict[str, str | int | None]:
        """Return a serialization-ready row in the canonical column order."""

        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "suffix": self.suffix,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "status": self.status.value,
            "error": self.error,
        }


def _resolved_root(root: str | Path) -> Path:
    candidate = Path(root)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"manifest root is unavailable: {candidate}") from exc
    if not resolved.is_dir():
        raise NotADirectoryError(f"manifest root is not a directory: {candidate}")
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _error_code(prefix: str, error: BaseException) -> str:
    return f"{prefix}:{type(error).__name__}"


@dataclass(frozen=True, slots=True)
class _Candidate:
    relative_path: str
    resolved_path: Path | None
    error: BaseException | None = None


def _iter_candidates(root: Path) -> Iterator[_Candidate]:
    """Walk regular directories while never traversing a directory symlink."""

    pending: list[tuple[Path, Path]] = [(root, Path())]
    while pending:
        directory, relative_directory = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            # A partial manifest must never look like a complete, clean scan.
            # Keep the public error path-free because directory names can carry
            # sensitive source identifiers.
            raise OSError(_error_code("DIRECTORY_SCAN_ERROR", exc)) from exc

        directories: list[tuple[Path, Path]] = []
        for entry in entries:
            path = Path(entry.path)
            relative = relative_directory / entry.name
            relative_text = relative.as_posix()
            try:
                if entry.is_symlink():
                    non_strict_target = path.resolve(strict=False)
                    if not _is_within(non_strict_target, root):
                        continue
                    try:
                        target = path.resolve(strict=True)
                    except OSError as exc:
                        yield _Candidate(relative_text, None, exc)
                        continue
                    if not _is_within(target, root):
                        continue
                    # Internal directory symlinks are intentionally not followed;
                    # this prevents cycles and duplicate traversal.
                    if target.is_file():
                        yield _Candidate(relative_text, target)
                    continue

                if entry.is_dir(follow_symlinks=False):
                    resolved_directory = path.resolve(strict=True)
                    if _is_within(resolved_directory, root):
                        directories.append((resolved_directory, relative))
                    continue
                if entry.is_file(follow_symlinks=False):
                    resolved_file = path.resolve(strict=True)
                    if _is_within(resolved_file, root):
                        yield _Candidate(relative_text, resolved_file)
            except OSError as exc:
                yield _Candidate(relative_text, None, exc)

        # Reverse push preserves lexicographic depth-first traversal.  The final
        # public sort is still authoritative and independent of filesystem order.
        pending.extend(reversed(directories))


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 hex digest of one file using bounded memory."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pillow() -> Any:
    try:
        return importlib.import_module("PIL.Image")
    except ModuleNotFoundError as exc:
        raise ManifestDependencyError(
            "image metadata was requested but Pillow is not installed"
        ) from exc


def _image_metadata(image_module: Any, path: Path) -> tuple[int, int, int]:
    with image_module.open(path) as image:
        width, height = image.size
        channels = len(image.getbands())
        image.verify()
    return int(width), int(height), int(channels)


def build_manifest(
    root: str | Path,
    *,
    include_image_metadata: bool = False,
    image_suffixes: Iterable[str] = DEFAULT_IMAGE_SUFFIXES,
) -> list[ManifestRecord]:
    """Recursively build a deterministic manifest bounded by ``root``.

    Symlinks resolving outside ``root`` are excluded without opening their
    targets.  Internal file symlinks are allowed; directory symlinks are not
    traversed.  Hashing remains enabled when image decoding reports corruption.
    """

    resolved_root = _resolved_root(root)
    normalized_image_suffixes = frozenset(
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        for suffix in image_suffixes
    )
    image_module: Any | None = None
    records: list[ManifestRecord] = []

    for candidate in _iter_candidates(resolved_root):
        suffix = Path(candidate.relative_path).suffix.lower()
        if candidate.resolved_path is None:
            error = candidate.error or OSError("unavailable file")
            records.append(
                ManifestRecord(
                    relative_path=candidate.relative_path,
                    size_bytes=None,
                    suffix=suffix,
                    sha256=None,
                    status=ManifestStatus.FILE_READ_ERROR,
                    error=_error_code("FILE_READ_ERROR", error),
                )
            )
            continue

        try:
            size = candidate.resolved_path.stat().st_size
            digest = sha256_file(candidate.resolved_path)
        except OSError as exc:
            records.append(
                ManifestRecord(
                    relative_path=candidate.relative_path,
                    size_bytes=None,
                    suffix=suffix,
                    sha256=None,
                    status=ManifestStatus.FILE_READ_ERROR,
                    error=_error_code("FILE_READ_ERROR", exc),
                )
            )
            continue

        width: int | None = None
        height: int | None = None
        channels: int | None = None
        status = ManifestStatus.OK
        error_text: str | None = None
        if include_image_metadata and suffix in normalized_image_suffixes:
            if image_module is None:
                image_module = _load_pillow()
            try:
                width, height, channels = _image_metadata(image_module, candidate.resolved_path)
            except (OSError, SyntaxError, ValueError) as exc:
                status = ManifestStatus.CORRUPT_IMAGE
                error_text = _error_code("IMAGE_DECODE_ERROR", exc)
        records.append(
            ManifestRecord(
                relative_path=candidate.relative_path,
                size_bytes=int(size),
                suffix=suffix,
                sha256=digest,
                width=width,
                height=height,
                channels=channels,
                status=status,
                error=error_text,
            )
        )

    return sorted(records, key=lambda record: record.relative_path)


def _validate_relative_path(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("manifest relative_path must be a string")
    if not value or value == "." or "\x00" in value:
        raise ValueError("manifest relative_path must name a relative file")
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ValueError("manifest relative_path must not be absolute")
    if ".." in posix_path.parts or ".." in windows_path.parts:
        raise ValueError("manifest relative_path must be root-contained")
    return value


def _sorted_records(records: Iterable[ManifestRecord]) -> list[ManifestRecord]:
    materialized = list(records)
    paths = [_validate_relative_path(record.relative_path) for record in materialized]
    if len(paths) != len(set(paths)):
        raise ValueError("manifest records must have unique relative_path values")
    return sorted(materialized, key=lambda record: record.relative_path)


def _write_csv(records: Sequence[ManifestRecord], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(MANIFEST_COLUMNS),
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(record.to_row() for record in records)


def _write_parquet(records: Sequence[ManifestRecord], output_path: Path) -> None:
    try:
        pyarrow: Any = importlib.import_module("pyarrow")
        parquet: Any = importlib.import_module("pyarrow.parquet")
    except ModuleNotFoundError as exc:
        raise ManifestDependencyError(
            "Parquet output was requested but pyarrow is not installed"
        ) from exc

    schema = pyarrow.schema(
        [
            pyarrow.field("relative_path", pyarrow.string(), nullable=False),
            pyarrow.field("size_bytes", pyarrow.int64(), nullable=True),
            pyarrow.field("suffix", pyarrow.string(), nullable=False),
            pyarrow.field("sha256", pyarrow.string(), nullable=True),
            pyarrow.field("width", pyarrow.int64(), nullable=True),
            pyarrow.field("height", pyarrow.int64(), nullable=True),
            pyarrow.field("channels", pyarrow.int64(), nullable=True),
            pyarrow.field("status", pyarrow.string(), nullable=False),
            pyarrow.field("error", pyarrow.string(), nullable=True),
        ]
    )
    table = pyarrow.Table.from_pylist([record.to_row() for record in records], schema=schema)
    parquet.write_table(
        table,
        output_path,
        compression=None,
        use_dictionary=False,
        write_statistics=False,
        data_page_version="1.0",
    )


def write_manifest(
    records: Iterable[ManifestRecord],
    output_path: str | Path,
    *,
    output_format: str | None = None,
) -> Path:
    """Write stable CSV or Parquet output, sorting records before serialization."""

    path = Path(output_path)
    selected_format = (output_format or path.suffix.removeprefix(".")).lower()
    if selected_format not in {"csv", "parquet", "pq"}:
        raise ValueError("manifest output format must be CSV or Parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_records = _sorted_records(records)
    if selected_format == "csv":
        _write_csv(sorted_records, path)
    else:
        _write_parquet(sorted_records, path)
    return path


def build_and_write_manifest(
    root: str | Path,
    output_path: str | Path,
    *,
    include_image_metadata: bool = False,
    image_suffixes: Iterable[str] = DEFAULT_IMAGE_SUFFIXES,
    output_format: str | None = None,
) -> list[ManifestRecord]:
    """Build a root-bounded manifest and serialize the same returned records."""

    resolved_root = _resolved_root(root)
    resolved_output = Path(output_path).resolve(strict=False)
    if _is_within(resolved_output, resolved_root):
        raise ValueError("manifest output must be outside the scanned root")

    records = build_manifest(
        resolved_root,
        include_image_metadata=include_image_metadata,
        image_suffixes=image_suffixes,
    )
    write_manifest(records, output_path, output_format=output_format)
    return records

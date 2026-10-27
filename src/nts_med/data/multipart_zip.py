"""Read-only verification for a ZIP split into sequential byte segments.

The verifier never joins parts on disk and never extracts or mutates source
files.  Its public result contains only aggregate counts and caller-supplied
part labels, so local raw-data paths need not enter repository evidence.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import zipfile
from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Final

_BUFFER_SIZE: Final[int] = 8 * 1024 * 1024
_WINDOWS_DRIVE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z]:")


@dataclass(frozen=True, slots=True)
class MultipartPartIdentity:
    """Path-free identity for one sequential archive part."""

    label: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class MultipartZipVerification:
    """Aggregate verification outcome suitable for privacy-safe evidence."""

    parts: tuple[MultipartPartIdentity, ...]
    combined_size: int
    combined_sha256: str
    archive_entry_count: int
    archive_file_count: int
    archive_file_bytes: int
    encrypted_entry_count: int
    unsafe_path_count: int
    crc_ok: bool
    crc_failure_member: str | None
    extracted_file_count: int
    matched_file_count: int
    missing_file_count: int
    extra_file_count: int
    size_mismatch_count: int

    @property
    def extraction_exact(self) -> bool:
        """Whether the extracted tree exactly matches archive file paths/sizes."""

        return (
            self.archive_file_count == self.extracted_file_count
            and self.matched_file_count == self.archive_file_count
            and self.missing_file_count == 0
            and self.extra_file_count == 0
            and self.size_mismatch_count == 0
        )


class MultipartReader(io.RawIOBase):
    """Seekable read-only view over sequential byte segments."""

    def __init__(self, part_paths: Iterable[str | Path]) -> None:
        super().__init__()
        self._paths = tuple(Path(path) for path in part_paths)
        if not self._paths:
            raise ValueError("at least one multipart segment is required")
        if any(not path.is_file() for path in self._paths):
            raise FileNotFoundError("every multipart segment must be an existing file")
        self._sizes = tuple(path.stat().st_size for path in self._paths)
        if any(size <= 0 for size in self._sizes):
            raise ValueError("multipart segments must be non-empty")
        starts: list[int] = []
        offset = 0
        for size in self._sizes:
            starts.append(offset)
            offset += size
        self._starts = tuple(starts)
        self._total_size = offset
        self._position = 0
        self._open_index: int | None = None
        self._handle: BinaryIO | None = None

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        self._checkClosed()
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        self._checkClosed()
        if whence == os.SEEK_SET:
            target = offset
        elif whence == os.SEEK_CUR:
            target = self._position + offset
        elif whence == os.SEEK_END:
            target = self._total_size + offset
        else:
            raise ValueError("unsupported seek origin")
        if target < 0:
            raise ValueError("negative seek position")
        self._position = target
        return target

    def _select_part(self, index: int) -> BinaryIO:
        if self._open_index != index:
            if self._handle is not None:
                self._handle.close()
            self._handle = self._paths[index].open("rb")
            self._open_index = index
        assert self._handle is not None
        return self._handle

    def read(self, size: int = -1) -> bytes:
        self._checkClosed()
        if self._position >= self._total_size or size == 0:
            return b""
        remaining = self._total_size - self._position
        wanted = remaining if size is None or size < 0 else min(size, remaining)
        output = bytearray()
        while wanted > 0:
            index = bisect_right(self._starts, self._position) - 1
            local_offset = self._position - self._starts[index]
            available = self._sizes[index] - local_offset
            take = min(wanted, available)
            handle = self._select_part(index)
            handle.seek(local_offset)
            chunk = handle.read(take)
            if len(chunk) != take:
                raise OSError("multipart segment ended before its recorded size")
            output.extend(chunk)
            self._position += take
            wanted -= take
        return bytes(output)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        super().close()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _multipart_identities(
    paths: tuple[Path, ...], labels: tuple[str, ...]
) -> tuple[tuple[MultipartPartIdentity, ...], str]:
    if len(paths) != len(labels):
        raise ValueError("part labels must exactly cover multipart segments")
    if len(labels) != len(set(labels)) or any(not label for label in labels):
        raise ValueError("part labels must be unique non-empty strings")
    combined = hashlib.sha256()
    parts: list[MultipartPartIdentity] = []
    for path, label in zip(paths, labels, strict=True):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(_BUFFER_SIZE):
                digest.update(chunk)
                combined.update(chunk)
        parts.append(MultipartPartIdentity(label, path.stat().st_size, digest.hexdigest()))
    return tuple(parts), combined.hexdigest()


def _unsafe_member(name: str) -> bool:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    return (
        normalized.startswith("/")
        or bool(_WINDOWS_DRIVE.match(normalized))
        or ".." in path.parts
    )


def _strip_archive_root(name: str, archive_root: str | None) -> str:
    normalized = name.replace("\\", "/").lstrip("/")
    if archive_root is None:
        return normalized
    prefix = archive_root.strip("/") + "/"
    if not normalized.startswith(prefix):
        raise ValueError(f"archive member is outside the declared root: {name!r}")
    return normalized[len(prefix) :]


def verify_multipart_zip(
    part_paths: Iterable[str | Path],
    *,
    part_labels: Iterable[str],
    extracted_root: str | Path,
    archive_root: str | None,
) -> MultipartZipVerification:
    """Verify hashes, ZIP CRC/path safety, and an existing extraction read-only."""

    paths = tuple(Path(path) for path in part_paths)
    labels = tuple(part_labels)
    reader = MultipartReader(paths)
    parts, combined_sha256 = _multipart_identities(paths, labels)
    archive_files: dict[str, int] = {}
    archive_entry_count = 0
    encrypted_entry_count = 0
    unsafe_path_count = 0
    crc_failure_member: str | None = None
    try:
        with zipfile.ZipFile(reader) as archive:
            infos = archive.infolist()
            archive_entry_count = len(infos)
            encrypted_entry_count = sum(bool(info.flag_bits & 0x1) for info in infos)
            unsafe_path_count = sum(_unsafe_member(info.filename) for info in infos)
            for info in infos:
                if info.is_dir():
                    continue
                relative = _strip_archive_root(info.filename, archive_root)
                if not relative or relative in archive_files:
                    raise ValueError("archive paths must be unique after root normalization")
                archive_files[relative] = info.file_size
            crc_failure_member = archive.testzip()
    finally:
        reader.close()

    root = Path(extracted_root)
    if not root.is_dir():
        raise FileNotFoundError("extracted_root must be an existing directory")
    extracted_files = {
        path.relative_to(root).as_posix(): path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
    }
    archive_names = set(archive_files)
    extracted_names = set(extracted_files)
    common = archive_names & extracted_names
    size_mismatch_count = sum(
        archive_files[name] != extracted_files[name] for name in common
    )
    return MultipartZipVerification(
        parts=parts,
        combined_size=sum(part.size for part in parts),
        combined_sha256=combined_sha256,
        archive_entry_count=archive_entry_count,
        archive_file_count=len(archive_files),
        archive_file_bytes=sum(archive_files.values()),
        encrypted_entry_count=encrypted_entry_count,
        unsafe_path_count=unsafe_path_count,
        crc_ok=crc_failure_member is None,
        crc_failure_member=crc_failure_member,
        extracted_file_count=len(extracted_files),
        matched_file_count=len(common),
        missing_file_count=len(archive_names - extracted_names),
        extra_file_count=len(extracted_names - archive_names),
        size_mismatch_count=size_mismatch_count,
    )


__all__ = [
    "MultipartPartIdentity",
    "MultipartReader",
    "MultipartZipVerification",
    "verify_multipart_zip",
]

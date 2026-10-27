"""Deterministic, local-only fingerprints for duplicate-image auditing.

This module deliberately stops before frozen-feature similarity: no model,
weight download, or network access is used here.  Exact byte hashes,
canonical decoded-pixel hashes, dHash, and pHash are complementary audit
signals and are reported as *candidates*, not automatic deletion decisions.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image, ImageOps

from nts_med.data.manifests import sha256_file

_NORMALIZED_PIXEL_HASH_VERSION: Final[bytes] = b"nts-med-normalized-rgb-v1\0"


class FingerprintMethod(StrEnum):
    """Supported deterministic duplicate-audit signals."""

    SHA256 = "sha256"
    NORMALIZED_PIXEL_SHA256 = "normalized_pixel_sha256"
    DHASH = "dhash"
    PHASH = "phash"


@dataclass(frozen=True, slots=True)
class ImageFingerprint:
    """Path-free fingerprints for one non-patient instance identifier."""

    instance_id: str
    sha256: str
    normalized_pixel_sha256: str | None = None
    dhash: str | None = None
    phash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.instance_id, str) or not self.instance_id:
            raise ValueError("instance_id must be a non-empty string")
        _validate_hex(self.sha256, name="sha256", expected_length=64)
        if self.normalized_pixel_sha256 is not None:
            _validate_hex(
                self.normalized_pixel_sha256,
                name="normalized_pixel_sha256",
                expected_length=64,
            )
        if self.dhash is not None:
            _validate_hex(self.dhash, name="dhash")
        if self.phash is not None:
            _validate_hex(self.phash, name="phash")


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    """Two or more instances sharing an exact fingerprint value."""

    method: FingerprintMethod
    fingerprint: str
    instance_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.instance_ids) < 2:
            raise ValueError("a duplicate group must contain at least two instances")
        if self.instance_ids != tuple(sorted(set(self.instance_ids))):
            raise ValueError("duplicate-group instance_ids must be unique and sorted")


@dataclass(frozen=True, slots=True)
class CrossSplitDuplicateCandidate:
    """A possible duplicate whose instances occur in different existing splits."""

    left_id: str
    right_id: str
    left_split: str
    right_split: str
    matched_methods: tuple[FingerprintMethod, ...]
    dhash_distance: int | None
    phash_distance: int | None

    def __post_init__(self) -> None:
        if self.left_id >= self.right_id:
            raise ValueError("candidate identifiers must be in strict sorted order")
        if self.left_split == self.right_split:
            raise ValueError("cross-split candidates must name different splits")
        if not self.matched_methods:
            raise ValueError("a duplicate candidate needs at least one matched method")


def _validate_hex(value: object, *, name: str, expected_length: int | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty hexadecimal string")
    if expected_length is not None and len(value) != expected_length:
        raise ValueError(f"{name} must contain exactly {expected_length} hexadecimal digits")
    if len(value) % 2:
        raise ValueError(f"{name} must contain an even number of hexadecimal digits")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be hexadecimal") from exc
    if value != value.lower():
        raise ValueError(f"{name} must use lowercase hexadecimal")
    return value


def _require_hash_size(value: object, *, name: str = "hash_size") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 2:
        raise ValueError(f"{name} must be at least 2")
    return value


def _bits_to_hex(bits: np.ndarray[tuple[int, ...], np.dtype[np.bool_]]) -> str:
    flattened = np.asarray(bits, dtype=np.uint8).reshape(-1)
    bit_count = int(flattened.size)
    if bit_count == 0 or bit_count % 4:
        raise ValueError("perceptual hashes require a positive multiple of four bits")
    value = 0
    for bit in flattened:
        value = (value << 1) | int(bit)
    return f"{value:0{bit_count // 4}x}"


def _canonical_rgb(image: Image.Image) -> Image.Image:
    transposed = ImageOps.exif_transpose(image)
    frame_count = int(getattr(transposed, "n_frames", 1))
    if frame_count != 1:
        raise ValueError("multi-frame images are not supported by the duplicate audit")
    return transposed.convert("RGB")


def normalized_pixel_sha256(image: Image.Image) -> str:
    """Hash canonical orientation, dimensions, and decoded RGB pixels.

    File container, compression level, and metadata are intentionally ignored.
    Image dimensions remain part of the digest so reshaped byte streams cannot
    collide merely because their flattened RGB bytes happen to match.
    """

    canonical = _canonical_rgb(image)
    digest = hashlib.sha256()
    digest.update(_NORMALIZED_PIXEL_HASH_VERSION)
    digest.update(struct.pack(">QQ", canonical.width, canonical.height))
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def difference_hash(image: Image.Image, *, hash_size: int = 8) -> str:
    """Return a fixed-size horizontal dHash using deterministic Pillow resize."""

    size = _require_hash_size(hash_size)
    canonical = _canonical_rgb(image).convert("L")
    resized = canonical.resize((size + 1, size), resample=Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.uint8)
    return _bits_to_hex(pixels[:, 1:] > pixels[:, :-1])


def _dct_matrix(size: int) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    positions = np.arange(size, dtype=np.float64) + 0.5
    frequencies = np.arange(size, dtype=np.float64)[:, None]
    matrix = np.cos((math.pi / size) * frequencies * positions)
    matrix[0, :] *= math.sqrt(1.0 / size)
    matrix[1:, :] *= math.sqrt(2.0 / size)
    return matrix


def perceptual_hash(
    image: Image.Image,
    *,
    hash_size: int = 8,
    high_frequency_factor: int = 4,
) -> str:
    """Return a DCT pHash without SciPy or model weights."""

    size = _require_hash_size(hash_size)
    factor = _require_hash_size(high_frequency_factor, name="high_frequency_factor")
    transform_size = size * factor
    canonical = _canonical_rgb(image).convert("L")
    resized = canonical.resize(
        (transform_size, transform_size),
        resample=Image.Resampling.LANCZOS,
    )
    pixels = np.asarray(resized, dtype=np.float64)
    matrix = _dct_matrix(transform_size)
    transformed = matrix @ pixels @ matrix.T
    low_frequencies = transformed[:size, :size]
    median = float(np.median(low_frequencies))
    return _bits_to_hex(low_frequencies > median)


def fingerprint_file(instance_id: str, path: str | Path) -> ImageFingerprint:
    """Hash one arbitrary local file without attempting image decoding."""

    return ImageFingerprint(instance_id=instance_id, sha256=sha256_file(path))


def fingerprint_image(
    instance_id: str,
    path: str | Path,
    *,
    hash_size: int = 8,
    phash_high_frequency_factor: int = 4,
) -> ImageFingerprint:
    """Compute all local duplicate-audit fingerprints for one image."""

    resolved_path = Path(path)
    byte_digest = sha256_file(resolved_path)
    with Image.open(resolved_path) as image:
        canonical = _canonical_rgb(image)
        pixel_digest = normalized_pixel_sha256(canonical)
        dhash_value = difference_hash(canonical, hash_size=hash_size)
        phash_value = perceptual_hash(
            canonical,
            hash_size=hash_size,
            high_frequency_factor=phash_high_frequency_factor,
        )
    return ImageFingerprint(
        instance_id=instance_id,
        sha256=byte_digest,
        normalized_pixel_sha256=pixel_digest,
        dhash=dhash_value,
        phash=phash_value,
    )


def _unique_fingerprints(
    fingerprints: Iterable[ImageFingerprint],
) -> tuple[ImageFingerprint, ...]:
    materialized = tuple(sorted(fingerprints, key=lambda item: item.instance_id))
    identifiers = [item.instance_id for item in materialized]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("fingerprints must have unique instance_id values")
    return materialized


def _fingerprint_value(item: ImageFingerprint, method: FingerprintMethod) -> str | None:
    if method is FingerprintMethod.SHA256:
        return item.sha256
    if method is FingerprintMethod.NORMALIZED_PIXEL_SHA256:
        return item.normalized_pixel_sha256
    if method is FingerprintMethod.DHASH:
        return item.dhash
    return item.phash


def group_fingerprints(
    fingerprints: Iterable[ImageFingerprint],
    method: FingerprintMethod,
) -> tuple[DuplicateGroup, ...]:
    """Group exact equality for one fingerprint method, omitting singletons."""

    if not isinstance(method, FingerprintMethod):
        raise TypeError("method must be a FingerprintMethod")
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for item in _unique_fingerprints(fingerprints):
        value = _fingerprint_value(item, method)
        if value is not None:
            grouped[value].append(item.instance_id)
    return tuple(
        DuplicateGroup(method, value, tuple(sorted(instance_ids)))
        for value, instance_ids in sorted(grouped.items())
        if len(instance_ids) >= 2
    )


def group_sha256(
    fingerprints: Iterable[ImageFingerprint],
) -> tuple[DuplicateGroup, ...]:
    """Return exact byte-level duplicate groups."""

    return group_fingerprints(fingerprints, FingerprintMethod.SHA256)


def group_normalized_pixels(
    fingerprints: Iterable[ImageFingerprint],
) -> tuple[DuplicateGroup, ...]:
    """Return exact decoded-pixel duplicate groups."""

    return group_fingerprints(fingerprints, FingerprintMethod.NORMALIZED_PIXEL_SHA256)


def hamming_distance(left: str, right: str) -> int:
    """Return bit Hamming distance between same-width hexadecimal hashes."""

    left_value = _validate_hex(left, name="left")
    right_value = _validate_hex(right, name="right")
    if len(left_value) != len(right_value):
        raise ValueError("perceptual hashes must have equal widths")
    return (int(left_value, 16) ^ int(right_value, 16)).bit_count()


def _require_distance(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def find_cross_split_duplicate_candidates(
    fingerprints: Iterable[ImageFingerprint],
    split_by_instance: Mapping[str, str],
    *,
    dhash_max_distance: int = 0,
    phash_max_distance: int = 0,
) -> tuple[CrossSplitDuplicateCandidate, ...]:
    """Return deterministic candidate pairs spanning different existing splits.

    Exact SHA256 and normalized-pixel equality are always checked.  Perceptual
    thresholds are caller-controlled engineering settings; the neutral default
    of zero only accepts identical dHash/pHash bit strings.
    """

    dhash_limit = _require_distance(dhash_max_distance, name="dhash_max_distance")
    phash_limit = _require_distance(phash_max_distance, name="phash_max_distance")
    records = _unique_fingerprints(fingerprints)
    identifiers = {record.instance_id for record in records}
    if set(split_by_instance) != identifiers:
        missing = sorted(identifiers - set(split_by_instance))
        unexpected = sorted(set(split_by_instance) - identifiers)
        raise ValueError(
            "split mapping must exactly cover fingerprint instances "
            f"(missing={missing}, unexpected={unexpected})"
        )
    for instance_id, split in split_by_instance.items():
        if not isinstance(split, str) or not split:
            raise ValueError(f"split for {instance_id!r} must be a non-empty string")

    candidates: list[CrossSplitDuplicateCandidate] = []
    method_order = tuple(FingerprintMethod)
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            left_split = split_by_instance[left.instance_id]
            right_split = split_by_instance[right.instance_id]
            if left_split == right_split:
                continue

            matched: set[FingerprintMethod] = set()
            if left.sha256 == right.sha256:
                matched.add(FingerprintMethod.SHA256)
            if (
                left.normalized_pixel_sha256 is not None
                and left.normalized_pixel_sha256 == right.normalized_pixel_sha256
            ):
                matched.add(FingerprintMethod.NORMALIZED_PIXEL_SHA256)

            dhash_distance: int | None = None
            if left.dhash is not None and right.dhash is not None:
                dhash_distance = hamming_distance(left.dhash, right.dhash)
                if dhash_distance <= dhash_limit:
                    matched.add(FingerprintMethod.DHASH)

            phash_distance: int | None = None
            if left.phash is not None and right.phash is not None:
                phash_distance = hamming_distance(left.phash, right.phash)
                if phash_distance <= phash_limit:
                    matched.add(FingerprintMethod.PHASH)

            if matched:
                candidates.append(
                    CrossSplitDuplicateCandidate(
                        left_id=left.instance_id,
                        right_id=right.instance_id,
                        left_split=left_split,
                        right_split=right_split,
                        matched_methods=tuple(
                            method for method in method_order if method in matched
                        ),
                        dhash_distance=dhash_distance,
                        phash_distance=phash_distance,
                    )
                )
    return tuple(candidates)


__all__ = [
    "CrossSplitDuplicateCandidate",
    "DuplicateGroup",
    "FingerprintMethod",
    "ImageFingerprint",
    "difference_hash",
    "find_cross_split_duplicate_candidates",
    "fingerprint_file",
    "fingerprint_image",
    "group_fingerprints",
    "group_normalized_pixels",
    "group_sha256",
    "hamming_distance",
    "normalized_pixel_sha256",
    "perceptual_hash",
]

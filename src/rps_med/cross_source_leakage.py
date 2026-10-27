"""Aggregate-only image overlap audit for TJDR and DDR validation."""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections import Counter
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from nts_med.data.adapters import AdapterStatus, DDRAdapter
from rps_med.tjdr_io import _index_split


@dataclass(frozen=True, slots=True)
class SourceFingerprints:
    """In-memory image fingerprints; individual values are never serialized."""

    encoded_exact: tuple[str, ...]
    pixel_exact: tuple[str, ...]
    dhash: tuple[int, ...]
    descriptors: np.ndarray[Any, np.dtype[np.float32]]
    sizes: Counter[str]

    @property
    def count(self) -> int:
        return len(self.encoded_exact)


def _pixel_digest(rgb: np.ndarray[Any, np.dtype[np.uint8]]) -> str:
    digest = hashlib.sha256()
    digest.update(str(rgb.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(rgb.tobytes(order="C"))
    return digest.hexdigest()


def _fingerprint_images(encoded_images: Iterable[bytes]) -> SourceFingerprints:
    encoded_exact: list[str] = []
    pixel_exact: list[str] = []
    dhash: list[int] = []
    descriptors: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    sizes: Counter[str] = Counter()
    for encoded in encoded_images:
        encoded_exact.append(hashlib.sha256(encoded).hexdigest())
        with Image.open(io.BytesIO(encoded)) as image:
            image.load()
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            sizes[f"{image.height}x{image.width}"] += 1
            gray9 = np.asarray(
                image.convert("L").resize((9, 8), Image.Resampling.LANCZOS),
                dtype=np.uint8,
            )
            gray64 = np.asarray(
                image.convert("L").resize((64, 64), Image.Resampling.LANCZOS),
                dtype=np.float32,
            )
        pixel_exact.append(_pixel_digest(rgb))
        comparison = gray9[:, 1:] > gray9[:, :-1]
        value = 0
        for bit in comparison.ravel():
            value = (value << 1) | int(bit)
        dhash.append(value)
        gradient = np.concatenate(
            (np.diff(gray64, axis=0).ravel(), np.diff(gray64, axis=1).ravel())
        )
        norm = float(np.linalg.norm(gradient))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("cross-source perceptual descriptor is degenerate")
        descriptors.append(np.asarray(gradient / norm, dtype=np.float32))
    if not descriptors:
        raise ValueError("cross-source leakage audit received no images")
    return SourceFingerprints(
        encoded_exact=tuple(encoded_exact),
        pixel_exact=tuple(pixel_exact),
        dhash=tuple(dhash),
        descriptors=np.stack(descriptors),
        sizes=sizes,
    )


def _tjdr_fingerprints(
    archive_paths: Sequence[str | Path], *, split: str
) -> SourceFingerprints:
    paths = tuple(Path(value) for value in archive_paths)
    with ExitStack() as stack:
        archives = tuple(stack.enter_context(zipfile.ZipFile(path)) for path in paths)
        pairs = _index_split(archives, split=split)
        encoded = []
        for pair in pairs:
            with archives[pair.image.archive_index].open(pair.image.member) as stream:
                encoded.append(stream.read())
    return _fingerprint_images(encoded)


def _ddr_valid_fingerprints(root: str | Path) -> SourceFingerprints:
    adapter = DDRAdapter(root)
    validation = adapter.validate()
    if validation.status is not AdapterStatus.READY:
        raise ValueError("DDR adapter must be ready")
    samples = tuple(sample for sample in adapter.samples() if sample.official_split == "valid")
    return _fingerprint_images(sample.image_path.read_bytes() for sample in samples)


def _within_exact(values: Sequence[str]) -> dict[str, int]:
    counts = Counter(values)
    groups = tuple(count for count in counts.values() if count > 1)
    return {
        "duplicate_pair_count": sum(count * (count - 1) // 2 for count in groups),
        "duplicate_group_count": len(groups),
        "affected_image_count": sum(groups),
    }


def _cross_exact(left: Sequence[str], right: Sequence[str]) -> dict[str, int]:
    left_counts = Counter(left)
    right_counts = Counter(right)
    shared = set(left_counts).intersection(right_counts)
    return {
        "duplicate_pair_count": sum(left_counts[key] * right_counts[key] for key in shared),
        "shared_content_group_count": len(shared),
        "affected_left_image_count": sum(left_counts[key] for key in shared),
        "affected_right_image_count": sum(right_counts[key] for key in shared),
    }


def _cross_hamming(
    left: Sequence[int], right: Sequence[int], thresholds: Sequence[int]
) -> dict[str, int]:
    output = {str(value): 0 for value in thresholds}
    for left_value in left:
        for right_value in right:
            distance = (left_value ^ right_value).bit_count()
            for threshold in thresholds:
                output[str(threshold)] += int(distance <= threshold)
    return output


def _cross_similarity(
    left: np.ndarray[Any, np.dtype[np.float32]],
    right: np.ndarray[Any, np.dtype[np.float32]],
    thresholds: Sequence[float],
) -> dict[str, dict[str, int]]:
    similarity = np.asarray(left @ right.T, dtype=np.float64)
    output: dict[str, dict[str, int]] = {}
    for threshold in thresholds:
        selected = similarity >= threshold
        left_indices, right_indices = np.nonzero(selected)
        output[f"{threshold:.4f}"] = {
            "pair_count": int(left_indices.size),
            "affected_left_image_count": int(np.unique(left_indices).size),
            "affected_right_image_count": int(np.unique(right_indices).size),
        }
    return output


def summarize_overlap(
    left: SourceFingerprints,
    right: SourceFingerprints,
    *,
    hamming_thresholds: Sequence[int] = (0, 2, 4, 6),
    gradient_thresholds: Sequence[float] = (0.99, 0.995, 0.999, 0.9999),
) -> dict[str, Any]:
    """Summarize image-level overlap without returning individual fingerprints."""

    return {
        "encoded_exact": _cross_exact(left.encoded_exact, right.encoded_exact),
        "decoded_pixel_exact": _cross_exact(left.pixel_exact, right.pixel_exact),
        "perceptual_candidate_pair_counts": _cross_hamming(
            left.dhash, right.dhash, hamming_thresholds
        ),
        "gradient_similarity_candidates": _cross_similarity(
            left.descriptors, right.descriptors, gradient_thresholds
        ),
    }


def audit_tjdr_ddr_model_overlap(
    tjdr_train_archives: Sequence[str | Path],
    tjdr_test_archives: Sequence[str | Path],
    ddr_root: str | Path,
) -> dict[str, Any]:
    """Audit DDR-valid overlap with prior TJDR development and test images."""

    train = _tjdr_fingerprints(tjdr_train_archives, split="train")
    test = _tjdr_fingerprints(tjdr_test_archives, split="test")
    ddr = _ddr_valid_fingerprints(ddr_root)
    return {
        "status": "RPS_MED_TJDR_DDR_MODEL_OVERLAP_AUDIT_COMPLETE",
        "method": {
            "encoded_exact": "SHA256_OF_ENCODED_IMAGE_BYTES",
            "decoded_pixel_exact": "SHA256_OF_RGB_SHAPE_AND_PIXELS",
            "perceptual": "64_BIT_HORIZONTAL_DIFFERENCE_HASH_GRAYSCALE_9X8",
            "strict_perceptual": "L2_NORMALIZED_64X64_GRAYSCALE_FIRST_DIFFERENCE_NCC",
            "candidate_status": "SCREENING_NOT_CONFIRMED_DUPLICATE",
        },
        "counts": {
            "tjdr_train": train.count,
            "tjdr_test": test.count,
            "ddr_valid": ddr.count,
        },
        "ddr_valid_internal": {
            "encoded_exact": _within_exact(ddr.encoded_exact),
            "decoded_pixel_exact": _within_exact(ddr.pixel_exact),
            "shape_counts": dict(sorted(ddr.sizes.items())),
        },
        "ddr_valid_vs_tjdr_train": summarize_overlap(ddr, train),
        "ddr_valid_vs_tjdr_test": summarize_overlap(ddr, test),
        "raw_images_persisted": False,
        "hash_rows_persisted": False,
        "local_paths_recorded": False,
        "source_identifiers_recorded": False,
        "patient_or_eye_identity_inference": False,
    }


__all__ = [
    "SourceFingerprints",
    "audit_tjdr_ddr_model_overlap",
    "summarize_overlap",
]

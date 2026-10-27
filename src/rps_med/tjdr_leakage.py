"""Aggregate-only exact and perceptual duplicate audit for TJDR images."""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections import Counter
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rps_med.tjdr_io import _index_split


@dataclass(frozen=True, slots=True)
class _Fingerprints:
    image_exact: tuple[str, ...]
    annotation_pixels: tuple[str, ...]
    dhash: tuple[int, ...]
    descriptors: np.ndarray[Any, np.dtype[np.float32]]
    sizes: Counter[str]


def _image_fingerprints(
    archive_paths: Sequence[str | Path], *, split: str
) -> _Fingerprints:
    paths = tuple(Path(path) for path in archive_paths)
    exact: list[str] = []
    annotation_pixels: list[str] = []
    dhash: list[int] = []
    descriptors: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    sizes: Counter[str] = Counter()
    with ExitStack() as stack:
        archives = tuple(stack.enter_context(zipfile.ZipFile(path)) for path in paths)
        pairs = _index_split(archives, split=split)
        for pair in pairs:
            with archives[pair.image.archive_index].open(pair.image.member) as stream:
                encoded = stream.read()
            exact.append(hashlib.sha256(encoded).hexdigest())
            with archives[pair.annotation.archive_index].open(
                pair.annotation.member
            ) as stream, Image.open(stream) as annotation:
                annotation.load()
                annotation_array = np.asarray(annotation)
            annotation_digest = hashlib.sha256()
            annotation_digest.update(str(annotation_array.shape).encode("ascii"))
            annotation_digest.update(b"\0")
            annotation_digest.update(annotation_array.dtype.str.encode("ascii"))
            annotation_digest.update(b"\0")
            annotation_digest.update(annotation_array.tobytes(order="C"))
            annotation_pixels.append(annotation_digest.hexdigest())
            with Image.open(io.BytesIO(encoded)) as image:
                image.load()
                sizes[f"{image.height}x{image.width}"] += 1
                gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
                array = np.asarray(gray, dtype=np.uint8)
            comparison = array[:, 1:] > array[:, :-1]
            value = 0
            for bit in comparison.ravel():
                value = (value << 1) | int(bit)
            dhash.append(value)
            with Image.open(io.BytesIO(encoded)) as image:
                image.load()
                gray64 = np.asarray(
                    image.convert("L").resize((64, 64), Image.Resampling.LANCZOS),
                    dtype=np.float32,
                )
            gradient = np.concatenate(
                (np.diff(gray64, axis=0).ravel(), np.diff(gray64, axis=1).ravel())
            )
            norm = float(np.linalg.norm(gradient))
            if not np.isfinite(norm) or norm <= 0.0:
                raise ValueError("TJDR perceptual descriptor is degenerate")
            descriptors.append(np.asarray(gradient / norm, dtype=np.float32))
    return _Fingerprints(
        image_exact=tuple(exact),
        annotation_pixels=tuple(annotation_pixels),
        dhash=tuple(dhash),
        descriptors=np.stack(descriptors),
        sizes=sizes,
    )


def _within_counts(values: Sequence[int], thresholds: Sequence[int]) -> dict[str, int]:
    counts = {str(threshold): 0 for threshold in thresholds}
    for left_index in range(len(values)):
        for right_index in range(left_index + 1, len(values)):
            distance = (values[left_index] ^ values[right_index]).bit_count()
            for threshold in thresholds:
                counts[str(threshold)] += int(distance <= threshold)
    return counts


def _cross_counts(
    left: Sequence[int], right: Sequence[int], thresholds: Sequence[int]
) -> dict[str, int]:
    counts = {str(threshold): 0 for threshold in thresholds}
    for left_value in left:
        for right_value in right:
            distance = (left_value ^ right_value).bit_count()
            for threshold in thresholds:
                counts[str(threshold)] += int(distance <= threshold)
    return counts


def _exact_within_summary(values: Sequence[str]) -> dict[str, int]:
    counts = Counter(values)
    duplicated = tuple(count for count in counts.values() if count > 1)
    return {
        "duplicate_pair_count": sum(count * (count - 1) // 2 for count in duplicated),
        "duplicate_group_count": len(duplicated),
        "affected_image_count": sum(duplicated),
    }


def _exact_cross_summary(left: Sequence[str], right: Sequence[str]) -> dict[str, int]:
    left_counts = Counter(left)
    right_counts = Counter(right)
    shared = set(left_counts).intersection(right_counts)
    return {
        "duplicate_pair_count": sum(left_counts[key] * right_counts[key] for key in shared),
        "shared_content_group_count": len(shared),
        "affected_train_image_count": sum(left_counts[key] for key in shared),
        "affected_test_image_count": sum(right_counts[key] for key in shared),
    }


def _annotation_agreement_within(
    image_hashes: Sequence[str], annotation_hashes: Sequence[str]
) -> dict[str, int]:
    matching = 0
    conflicting = 0
    for left_index in range(len(image_hashes)):
        for right_index in range(left_index + 1, len(image_hashes)):
            if image_hashes[left_index] != image_hashes[right_index]:
                continue
            if annotation_hashes[left_index] == annotation_hashes[right_index]:
                matching += 1
            else:
                conflicting += 1
    return {
        "matching_annotation_pixels_pair_count": matching,
        "conflicting_annotation_pixels_pair_count": conflicting,
    }


def _annotation_agreement_cross(
    left_image_hashes: Sequence[str],
    right_image_hashes: Sequence[str],
    left_annotation_hashes: Sequence[str],
    right_annotation_hashes: Sequence[str],
) -> dict[str, int]:
    matching = 0
    conflicting = 0
    for left_index, left_hash in enumerate(left_image_hashes):
        for right_index, right_hash in enumerate(right_image_hashes):
            if left_hash != right_hash:
                continue
            if left_annotation_hashes[left_index] == right_annotation_hashes[right_index]:
                matching += 1
            else:
                conflicting += 1
    return {
        "matching_annotation_pixels_pair_count": matching,
        "conflicting_annotation_pixels_pair_count": conflicting,
    }


def _similarity_summary(
    left: np.ndarray[Any, np.dtype[np.float32]],
    right: np.ndarray[Any, np.dtype[np.float32]],
    thresholds: Sequence[float],
    *,
    same_set: bool,
) -> dict[str, dict[str, int]]:
    similarity = np.asarray(left @ right.T, dtype=np.float64)
    if same_set:
        selector = np.triu(np.ones(similarity.shape, dtype=np.bool_), k=1)
    else:
        selector = np.ones(similarity.shape, dtype=np.bool_)
    output: dict[str, dict[str, int]] = {}
    for threshold in thresholds:
        selected = np.logical_and(selector, similarity >= threshold)
        left_indices, right_indices = np.nonzero(selected)
        output[f"{threshold:.4f}"] = {
            "pair_count": int(left_indices.size),
            "affected_left_image_count": int(np.unique(left_indices).size),
            "affected_right_image_count": int(np.unique(right_indices).size),
        }
    return output


def audit_tjdr_image_leakage(
    train_archives: Sequence[str | Path],
    test_archives: Sequence[str | Path],
    *,
    hamming_thresholds: Sequence[int] = (0, 2, 4, 6),
    gradient_similarity_thresholds: Sequence[float] = (0.99, 0.995, 0.999, 0.9999),
) -> dict[str, Any]:
    """Return path-free duplicate counts across the frozen official split."""

    thresholds = tuple(int(value) for value in hamming_thresholds)
    if not thresholds or tuple(sorted(set(thresholds))) != thresholds:
        raise ValueError("Hamming thresholds must be unique and increasing")
    if thresholds[0] < 0 or thresholds[-1] > 64:
        raise ValueError("Hamming thresholds must lie in [0, 64]")
    similarity_thresholds = tuple(float(value) for value in gradient_similarity_thresholds)
    if (
        not similarity_thresholds
        or tuple(sorted(set(similarity_thresholds))) != similarity_thresholds
        or similarity_thresholds[0] <= -1.0
        or similarity_thresholds[-1] > 1.0
    ):
        raise ValueError("gradient similarity thresholds must be increasing in (-1, 1]")
    train = _image_fingerprints(
        train_archives, split="train"
    )
    test = _image_fingerprints(
        test_archives, split="test"
    )
    train_exact_summary = _exact_within_summary(train.image_exact)
    train_exact_summary.update(
        _annotation_agreement_within(train.image_exact, train.annotation_pixels)
    )
    test_exact_summary = _exact_within_summary(test.image_exact)
    test_exact_summary.update(
        _annotation_agreement_within(test.image_exact, test.annotation_pixels)
    )
    cross_exact_summary = _exact_cross_summary(train.image_exact, test.image_exact)
    cross_exact_summary.update(
        _annotation_agreement_cross(
            train.image_exact,
            test.image_exact,
            train.annotation_pixels,
            test.annotation_pixels,
        )
    )
    return {
        "status": "TJDR_IMAGE_LEAKAGE_AGGREGATE_AUDIT_COMPLETE",
        "method": {
            "exact": "SHA256_OF_ENCODED_OFFICIAL_PNG_BYTES",
            "perceptual": "64_BIT_HORIZONTAL_DIFFERENCE_HASH_GRAYSCALE_9X8",
            "hamming_thresholds": list(thresholds),
            "strict_perceptual": "L2_NORMALIZED_64X64_GRAYSCALE_FIRST_DIFFERENCE_NCC",
            "gradient_similarity_thresholds": list(similarity_thresholds),
            "candidate_status": "SCREENING_NOT_CONFIRMED_DUPLICATE",
        },
        "train": {
            "image_count": len(train.image_exact),
            "shape_counts": dict(sorted(train.sizes.items())),
            "exact": train_exact_summary,
            "perceptual_candidate_pair_counts": _within_counts(train.dhash, thresholds),
            "gradient_similarity_candidates": _similarity_summary(
                train.descriptors,
                train.descriptors,
                similarity_thresholds,
                same_set=True,
            ),
        },
        "test": {
            "image_count": len(test.image_exact),
            "shape_counts": dict(sorted(test.sizes.items())),
            "exact": test_exact_summary,
            "perceptual_candidate_pair_counts": _within_counts(test.dhash, thresholds),
            "gradient_similarity_candidates": _similarity_summary(
                test.descriptors,
                test.descriptors,
                similarity_thresholds,
                same_set=True,
            ),
        },
        "cross_split": {
            "exact": cross_exact_summary,
            "perceptual_candidate_pair_counts": _cross_counts(
                train.dhash, test.dhash, thresholds
            ),
            "gradient_similarity_candidates": _similarity_summary(
                train.descriptors,
                test.descriptors,
                similarity_thresholds,
                same_set=False,
            ),
        },
        "raw_images_persisted": False,
        "hash_rows_persisted": False,
        "local_paths_recorded": False,
        "source_identifiers_recorded": False,
    }


__all__ = ["audit_tjdr_image_leakage"]

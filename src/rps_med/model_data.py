"""Leakage-controlled TJDR image cache for the RPS-Med model stage."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from nts_med.concepts.fov import FOVStatus, lesion_outside_fov_ratio
from nts_med.gate0.source_io import _burden
from nts_med.reproducibility.freeze import sha256_file
from rps_med.tjdr_io import (
    TJDR_AXES_TO_CLASSES,
    _index_split,
    _read_image,
    _validate_annotation,
    extract_tjdr_retinal_fov,
)

MODEL_AXES: Final = ("ma", "hem", "ex", "cws")


@dataclass(frozen=True, slots=True)
class _ContentGroup:
    image_hash: str
    encoded_image: bytes
    annotations: tuple[NDArray[Any], ...]
    source_shape: tuple[int, int]


def _encoded_image_hashes(
    archive_paths: Sequence[str | Path], *, split: str
) -> frozenset[str]:
    paths = tuple(Path(value) for value in archive_paths)
    with ExitStack() as stack:
        archives = tuple(stack.enter_context(zipfile.ZipFile(path)) for path in paths)
        pairs = _index_split(archives, split=split)
        hashes: set[str] = set()
        for pair in pairs:
            with archives[pair.image.archive_index].open(pair.image.member) as stream:
                hashes.add(hashlib.sha256(stream.read()).hexdigest())
    return frozenset(hashes)


def _load_content_groups(
    archive_paths: Sequence[str | Path],
    *,
    split: str,
    excluded_hashes: frozenset[str],
) -> tuple[tuple[_ContentGroup, ...], int, int]:
    paths = tuple(Path(value) for value in archive_paths)
    grouped: dict[str, tuple[bytes, list[NDArray[Any]], tuple[int, int]]] = {}
    excluded_count = 0
    input_count = 0
    with ExitStack() as stack:
        archives = tuple(stack.enter_context(zipfile.ZipFile(path)) for path in paths)
        pairs = _index_split(archives, split=split)
        input_count = len(pairs)
        for pair in pairs:
            with archives[pair.image.archive_index].open(pair.image.member) as stream:
                encoded = stream.read()
            image_hash = hashlib.sha256(encoded).hexdigest()
            if image_hash in excluded_hashes:
                excluded_count += 1
                continue
            image, _ = _read_image(archives, pair.image, rgb=True)
            annotation, _ = _read_image(archives, pair.annotation, rgb=False)
            _validate_annotation(annotation, image.shape[:2], allow_all_background=True)
            if image_hash in grouped:
                prior_image, annotations, prior_shape = grouped[image_hash]
                if encoded != prior_image or tuple(image.shape[:2]) != prior_shape:
                    raise ValueError("TJDR content-hash group is internally inconsistent")
                annotations.append(annotation)
            else:
                grouped[image_hash] = (
                    encoded,
                    [annotation],
                    (int(image.shape[0]), int(image.shape[1])),
                )
    groups = tuple(
        _ContentGroup(key, encoded, tuple(annotations), shape)
        for key, (encoded, annotations, shape) in sorted(grouped.items())
    )
    return groups, input_count, excluded_count


def _resize_rgb(encoded: bytes, image_size: int) -> NDArray[np.uint8]:
    with Image.open(io.BytesIO(encoded)) as image:
        image.load()
        resized = image.convert("RGB").resize(
            (image_size, image_size), Image.Resampling.LANCZOS
        )
        return np.asarray(resized, dtype=np.uint8).copy()


def _group_targets(
    group: _ContentGroup,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    with Image.open(io.BytesIO(group.encoded_image)) as image:
        image.load()
        rgb = np.asarray(image.convert("RGB")).copy()
    fov = extract_tjdr_retinal_fov(rgb)
    if fov.status is not FOVStatus.VALID or fov.mask is None:
        raise ValueError("TJDR model cache encountered an invalid FOV")
    burdens: list[list[float]] = []
    presences: list[list[float]] = []
    for annotation in group.annotations:
        masks = {
            axis: np.asarray(annotation == source_class, dtype=np.bool_)
            for axis, source_class in TJDR_AXES_TO_CLASSES.items()
        }
        if any(lesion_outside_fov_ratio(mask, fov.mask) > 0.0 for mask in masks.values()):
            raise ValueError("TJDR model-cache lesion lies outside the FOV")
        burden = _burden(masks, fov.mask)
        burdens.append([burden[axis] for axis in MODEL_AXES])
        presences.append([float(bool(mask.any())) for mask in masks.values()])
    return (
        np.asarray(np.mean(burdens, axis=0), dtype=np.float32),
        np.asarray(np.mean(presences, axis=0), dtype=np.float32),
    )


def _feature_vector(
    presence: NDArray[np.float32],
    shape: tuple[int, int],
    shape_levels: tuple[tuple[int, int], ...],
) -> NDArray[np.float64]:
    shape_one_hot = np.zeros(len(shape_levels), dtype=np.float64)
    shape_one_hot[shape_levels.index(shape)] = 1.0
    any_presence = np.asarray(presence > 0.0, dtype=np.float64)
    all_background = np.asarray([float(not bool(any_presence.any()))])
    return np.concatenate((any_presence, shape_one_hot, all_background))


def deterministic_multilabel_split(
    content_hashes: Sequence[str],
    features: NDArray[np.float64],
    *,
    validation_fraction: float,
    seed: int,
) -> NDArray[np.uint8]:
    """Select a deterministic validation subset matching aggregate feature counts."""

    hashes = tuple(content_hashes)
    if features.ndim != 2 or features.shape[0] != len(hashes):
        raise ValueError("split features do not match content groups")
    if len(hashes) < 10 or len(set(hashes)) != len(hashes):
        raise ValueError("split requires at least ten unique content groups")
    if not 0.05 <= validation_fraction <= 0.5:
        raise ValueError("validation fraction must lie in [0.05, 0.5]")
    target_count = round(len(hashes) * validation_fraction)
    target_features = features.sum(axis=0) * (target_count / len(hashes))
    scales = np.maximum(target_features, 1.0)
    priorities = tuple(
        int.from_bytes(
            hashlib.sha256(f"{seed}\0{value}".encode()).digest()[:8], "big"
        )
        for value in hashes
    )
    selected = np.zeros(len(hashes), dtype=np.bool_)
    current = np.zeros(features.shape[1], dtype=np.float64)
    for _ in range(target_count):
        best_index = -1
        best_key: tuple[float, int] | None = None
        for index in range(len(hashes)):
            if selected[index]:
                continue
            after = current + features[index]
            score = float(np.sum(np.square((after - target_features) / scales)))
            key = (score, priorities[index])
            if best_key is None or key < best_key:
                best_key = key
                best_index = index
        if best_index < 0:
            raise RuntimeError("deterministic split selection failed")
        selected[best_index] = True
        current += features[best_index]
    return np.asarray(selected, dtype=np.uint8)


def _count_by_partition(
    values: NDArray[np.float32], partition: NDArray[np.uint8]
) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for name, code in (("train", 0), ("validation", 1)):
        selected = values[partition == code]
        output[name] = {
            axis: int(np.count_nonzero(selected[:, index] > 0.0))
            for index, axis in enumerate(MODEL_AXES)
        }
    return output


def build_tjdr_development_cache(
    train_archives: Sequence[str | Path],
    test_archives: Sequence[str | Path],
    output: str | Path,
    *,
    image_size: int,
    validation_fraction: float,
    split_seed: int,
) -> dict[str, Any]:
    """Create a write-once, ignored cache after exact cross-split exclusion."""

    output_path = Path(output)
    temporary = output_path.with_suffix(".tmp.npz")
    if output_path.exists() or temporary.exists():
        raise FileExistsError("TJDR model cache output already exists")
    if image_size < 224 or image_size % 32:
        raise ValueError("image size must be at least 224 and divisible by 32")
    test_hashes = _encoded_image_hashes(test_archives, split="test")
    groups, input_count, excluded_count = _load_content_groups(
        train_archives,
        split="train",
        excluded_hashes=test_hashes,
    )
    images: list[NDArray[np.uint8]] = []
    burdens: list[NDArray[np.float32]] = []
    presences: list[NDArray[np.float32]] = []
    group_sizes: list[int] = []
    shapes: list[tuple[int, int]] = []
    for group in groups:
        burden, presence = _group_targets(group)
        images.append(_resize_rgb(group.encoded_image, image_size))
        burdens.append(burden)
        presences.append(presence)
        group_sizes.append(len(group.annotations))
        shapes.append(group.source_shape)
    burden_array = np.stack(burdens)
    presence_array = np.stack(presences)
    shape_levels = tuple(sorted(set(shapes)))
    feature_array = np.stack(
        [
            _feature_vector(presence, shape, shape_levels)
            for presence, shape in zip(presence_array, shapes, strict=True)
        ]
    )
    partition = deterministic_multilabel_split(
        tuple(group.image_hash for group in groups),
        feature_array,
        validation_fraction=validation_fraction,
        seed=split_seed,
    )
    shape_codes = np.asarray([shape_levels.index(shape) for shape in shapes], dtype=np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        temporary,
        images=np.stack(images),
        burdens=burden_array,
        presence=presence_array,
        partition=partition,
        annotation_group_size=np.asarray(group_sizes, dtype=np.uint8),
        source_shape_code=shape_codes,
        axis_names=np.asarray(MODEL_AXES),
        image_size=np.asarray([image_size], dtype=np.int32),
    )
    temporary.replace(output_path)
    output_sha = sha256_file(output_path)
    partition_digest = hashlib.sha256(partition.tobytes()).hexdigest()
    group_counter = Counter(group_sizes)
    summary = {
        "status": "TJDR_DEVELOPMENT_MODEL_CACHE_COMPLETE",
        "input_train_image_count": input_count,
        "cross_test_exact_excluded_train_count": excluded_count,
        "unique_development_image_count": len(groups),
        "collapsed_annotation_group_size_counts": {
            str(key): group_counter[key] for key in sorted(group_counter)
        },
        "conflicting_duplicate_groups_aggregated": sum(size > 1 for size in group_sizes),
        "target_aggregation": "ARITHMETIC_MEAN_OF_TRANSFORMED_BURDENS_PER_EXACT_IMAGE",
        "train_count": int(np.count_nonzero(partition == 0)),
        "validation_count": int(np.count_nonzero(partition == 1)),
        "positive_content_counts": _count_by_partition(presence_array, partition),
        "source_shape_levels": [f"{height}x{width}" for height, width in shape_levels],
        "source_shape_counts": {
            name: {
                f"{height}x{width}": int(
                    np.count_nonzero(
                        np.logical_and(partition == code, shape_codes == shape_index)
                    )
                )
                for shape_index, (height, width) in enumerate(shape_levels)
            }
            for name, code in (("train", 0), ("validation", 1))
        },
        "image_size": image_size,
        "validation_fraction": validation_fraction,
        "split_seed": split_seed,
        "partition_digest": partition_digest,
        "cache_sha256": output_sha,
        "train_archive_sha256": [sha256_file(Path(value)) for value in train_archives],
        "test_archive_sha256": [sha256_file(Path(value)) for value in test_archives],
        "source_identifiers_recorded": False,
        "source_paths_recorded": False,
        "content_hash_rows_recorded": False,
        "official_test_pixels_decoded": False,
    }
    return summary


def load_tjdr_model_cache(path: str | Path) -> Mapping[str, NDArray[Any]]:
    """Load and validate the non-identifying arrays required for model development."""

    with np.load(Path(path), allow_pickle=False) as archive:
        values = {key: np.asarray(archive[key]).copy() for key in archive.files}
    required = {
        "images",
        "burdens",
        "presence",
        "partition",
        "annotation_group_size",
        "source_shape_code",
        "axis_names",
        "image_size",
    }
    if set(values) != required:
        raise ValueError("TJDR model cache schema is invalid")
    images = values["images"]
    burdens = values["burdens"]
    presence = values["presence"]
    partition = values["partition"]
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError("TJDR cache image array is invalid")
    if burdens.shape != (images.shape[0], 4) or presence.shape != burdens.shape:
        raise ValueError("TJDR cache target arrays are invalid")
    if partition.shape != (images.shape[0],) or not set(np.unique(partition)).issubset({0, 1}):
        raise ValueError("TJDR cache partition is invalid")
    if not np.isfinite(burdens).all() or bool((burdens < 0.0).any()):
        raise ValueError("TJDR cache burdens are invalid")
    if not np.isfinite(presence).all() or bool((presence < 0.0).any()) or bool(
        (presence > 1.0).any()
    ):
        raise ValueError("TJDR cache presence targets are invalid")
    return values


def load_tjdr_test_model_arrays(
    test_archives: Sequence[str | Path],
    *,
    expected_count: int,
    image_size: int,
) -> tuple[NDArray[np.uint8], NDArray[np.float32], dict[str, Any]]:
    """Decode the official test split in memory after formal authorization."""

    groups, input_count, excluded_count = _load_content_groups(
        test_archives,
        split="test",
        excluded_hashes=frozenset(),
    )
    if input_count != expected_count or excluded_count or len(groups) != expected_count:
        raise ValueError("TJDR official-test unique content boundary is invalid")
    if any(len(group.annotations) != 1 for group in groups):
        raise ValueError("TJDR official test contains duplicate image content")
    images: list[NDArray[np.uint8]] = []
    burdens: list[NDArray[np.float32]] = []
    for group in groups:
        burden, _ = _group_targets(group)
        images.append(_resize_rgb(group.encoded_image, image_size))
        burdens.append(burden)
    image_array = np.stack(images)
    burden_array = np.stack(burdens)
    return (
        image_array,
        burden_array,
        {
            "input_image_count": input_count,
            "unique_image_count": len(groups),
            "image_size": image_size,
            "source_identifiers_recorded": False,
            "source_paths_recorded": False,
            "test_cache_persisted": False,
        },
    )


def cache_summary_json(summary: Mapping[str, Any]) -> str:
    """Return a canonical aggregate-only representation for a write-once report."""

    return json.dumps(dict(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


__all__ = [
    "MODEL_AXES",
    "build_tjdr_development_cache",
    "cache_summary_json",
    "deterministic_multilabel_split",
    "load_tjdr_model_cache",
    "load_tjdr_test_model_arrays",
]

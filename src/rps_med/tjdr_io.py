"""Privacy-safe, read-only TJDR development input boundary.

The official test archive is deliberately outside every function in this
module.  Multi-part official-train ZIPs are indexed in memory and no source
member name or per-image measurement is returned or persisted.
"""

from __future__ import annotations

import math
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from nts_med.concepts.fov import FOVResult, FOVStatus, lesion_outside_fov_ratio
from nts_med.data.adapters import opaque_sample_id
from nts_med.gate0.execution import MeasurementRecord, SourceMeasurements
from nts_med.gate0.source_io import _burden
from nts_med.reproducibility.freeze import sha256_file
from rps_med.fov_v07 import extract_ddr_retinal_fov_v07

TJDR_FOV_ALGORITHM_VERSION: Final = "TJDR_BLACK_BACKGROUND_DOMINANT_COMPONENT_V1_CANDIDATE"
TJDR_AXES_TO_CLASSES: Final[Mapping[str, int]] = {
    "ma": 3,
    "hem": 2,
    "ex": 1,
    "cws": 4,
}
TJDR_ALLOWED_CLASSES: Final = frozenset({0, 1, 2, 3, 4})


class TJDRSchemaError(ValueError):
    """Format failure carrying only a stable, path-free reason code."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class TJDRStructuralBoundary:
    """Aggregate-only identity of one official TJDR split."""

    split: str
    archive_count: int
    archive_sha256: tuple[str, ...]
    combined_member_count: int
    image_count: int
    annotation_count: int
    opaque_pair_digest: str


@dataclass(frozen=True, slots=True)
class _MemberLocation:
    archive_index: int
    member: str


@dataclass(frozen=True, slots=True)
class _PairLocation:
    stem: str
    image: _MemberLocation
    annotation: _MemberLocation


def _opaque_digest(values: Sequence[str]) -> str:
    import hashlib

    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _index_split(
    archives: Sequence[zipfile.ZipFile], *, split: str
) -> tuple[_PairLocation, ...]:
    if split not in {"train", "test"}:
        raise ValueError("TJDR split must be train or test")
    members: set[str] = set()
    images: dict[str, _MemberLocation] = {}
    annotations: dict[str, _MemberLocation] = {}
    for archive_index, archive in enumerate(archives):
        for info in archive.infolist():
            if info.is_dir():
                continue
            member = info.filename.replace("\\", "/")
            path = PurePosixPath(member)
            if len(path.parts) != 3 or path.parts[0] != split:
                raise ValueError("TJDR archive contains an unexpected member boundary")
            role = path.parts[1]
            if role not in {"image", "annotation"} or path.suffix.casefold() != ".png":
                raise ValueError("TJDR archive member schema is invalid")
            canonical_member = member.casefold()
            if canonical_member in members:
                raise ValueError("TJDR multi-part archives contain a duplicate member")
            members.add(canonical_member)
            stem = path.stem.casefold()
            destination = images if role == "image" else annotations
            if stem in destination:
                raise ValueError("TJDR split contains a duplicate sample stem")
            destination[stem] = _MemberLocation(archive_index, member)
    if not images or set(images) != set(annotations):
        raise ValueError("TJDR image and annotation stems do not match")
    return tuple(
        _PairLocation(stem, images[stem], annotations[stem]) for stem in sorted(images)
    )


def inspect_tjdr_structure(
    archive_paths: Sequence[str | Path],
    *,
    split: str,
) -> TJDRStructuralBoundary:
    """Validate central-directory structure without decoding scientific pixels."""

    paths = tuple(Path(path) for path in archive_paths)
    if not paths:
        raise ValueError("at least one TJDR archive is required")
    with ExitStack() as stack:
        archives = tuple(stack.enter_context(zipfile.ZipFile(path)) for path in paths)
        pairs = _index_split(archives, split=split)
        member_count = sum(
            1 for archive in archives for info in archive.infolist() if not info.is_dir()
        )
    opaque_ids = tuple(opaque_sample_id("tjdr", f"{split}/{pair.stem}") for pair in pairs)
    return TJDRStructuralBoundary(
        split=split,
        archive_count=len(paths),
        archive_sha256=tuple(sha256_file(path) for path in paths),
        combined_member_count=member_count,
        image_count=len(pairs),
        annotation_count=len(pairs),
        opaque_pair_digest=_opaque_digest(opaque_ids),
    )


def _read_image(
    archives: Sequence[zipfile.ZipFile], location: _MemberLocation, *, rgb: bool
) -> tuple[NDArray[Any], str]:
    with archives[location.archive_index].open(location.member) as stream, Image.open(
        stream
    ) as image:
        image.load()
        mode = image.mode
        value = image.convert("RGB") if rgb else image
        return np.asarray(value).copy(), mode


def extract_tjdr_retinal_fov(
    image: NDArray[Any],
    *,
    background_threshold: float = 5.0,
    minimum_area_ratio: float = 0.10,
    maximum_area_ratio: float = 0.98,
    substantial_component_ratio: float = 0.001,
    dominant_component_min_share: float = 0.90,
) -> FOVResult:
    """Apply the inherited image-only FOV candidate with a TJDR version label."""

    result = extract_ddr_retinal_fov_v07(
        image,
        background_threshold=background_threshold,
        minimum_area_ratio=minimum_area_ratio,
        maximum_area_ratio=maximum_area_ratio,
        substantial_component_ratio=substantial_component_ratio,
        dominant_component_min_share=dominant_component_min_share,
    )
    return FOVResult(
        status=result.status,
        mask=result.mask,
        foreground_component_count=result.foreground_component_count,
        substantial_component_count=result.substantial_component_count,
        area_ratio=result.area_ratio,
        border_contact=result.border_contact,
        method=result.method,
        algorithm_version=TJDR_FOV_ALGORITHM_VERSION,
    )


def _validate_annotation(
    mask: NDArray[Any],
    image_shape: Sequence[int],
    *,
    allow_all_background: bool,
) -> set[int]:
    if len(image_shape) != 2 or mask.ndim != 2 or tuple(mask.shape) != tuple(image_shape):
        raise TJDRSchemaError("ANNOTATION_DIMENSION_MISMATCH")
    if not np.issubdtype(mask.dtype, np.integer):
        raise TJDRSchemaError("ANNOTATION_NONINTEGER_DTYPE")
    observed = {int(value) for value in np.unique(mask)}
    if not observed.issubset(TJDR_ALLOWED_CLASSES):
        raise TJDRSchemaError("ANNOTATION_UNKNOWN_CLASS")
    if not allow_all_background and not observed.intersection({1, 2, 3, 4}):
        raise TJDRSchemaError("ANNOTATION_NO_LESION_CLASS")
    return observed


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("minimum", "q25", "median", "q75", "maximum")}
    result = np.quantile(np.asarray(values, dtype=np.float64), (0, 0.25, 0.5, 0.75, 1))
    return {
        name: float(value)
        for name, value in zip(
            ("minimum", "q25", "median", "q75", "maximum"), result, strict=True
        )
    }


def audit_tjdr_train(
    archive_paths: Sequence[str | Path],
    *,
    expected_count: int,
    background_threshold: float = 5.0,
    minimum_area_ratio: float = 0.10,
    maximum_area_ratio: float = 0.98,
    substantial_component_ratio: float = 0.001,
    dominant_component_min_share: float = 0.90,
    allow_all_background: bool = False,
) -> dict[str, Any]:
    """Decode official train and return only fixed, aggregate-safe audit facts."""

    paths = tuple(Path(path) for path in archive_paths)
    reason_counts: Counter[str] = Counter()
    image_modes: Counter[str] = Counter()
    annotation_modes: Counter[str] = Counter()
    image_shapes: Counter[str] = Counter()
    annotation_value_counts: Counter[int] = Counter()
    positive_image_counts: Counter[str] = Counter()
    fov_status_counts: Counter[str] = Counter()
    fov_method_counts: Counter[str] = Counter()
    outside_fov_image_counts: Counter[str] = Counter()
    outside_fov_pixel_counts: Counter[str] = Counter()
    area_ratios: list[float] = []
    all_background_count = 0
    format_valid = 0
    decoded = 0

    with ExitStack() as stack:
        archives = tuple(stack.enter_context(zipfile.ZipFile(path)) for path in paths)
        pairs = _index_split(archives, split="train")
        if len(pairs) != expected_count:
            reason_counts["TRAIN_PAIR_COUNT_MISMATCH"] += 1
        for pair in pairs:
            try:
                try:
                    image, image_mode = _read_image(archives, pair.image, rgb=True)
                except (OSError, TypeError, ValueError):
                    reason_counts["IMAGE_DECODE_ERROR"] += 1
                    continue
                try:
                    annotation, annotation_mode = _read_image(
                        archives, pair.annotation, rgb=False
                    )
                except (OSError, TypeError, ValueError):
                    reason_counts["ANNOTATION_DECODE_ERROR"] += 1
                    continue
                if image.ndim != 3 or image.shape[2] != 3:
                    reason_counts["IMAGE_SCHEMA_ERROR"] += 1
                    continue
                observed = _validate_annotation(
                    annotation,
                    image.shape[:2],
                    allow_all_background=allow_all_background,
                )
                if observed == {0}:
                    all_background_count += 1
                for value in observed:
                    annotation_value_counts[value] += 1
                for axis, source_class in TJDR_AXES_TO_CLASSES.items():
                    if bool((annotation == source_class).any()):
                        positive_image_counts[axis] += 1
                image_modes[image_mode] += 1
                annotation_modes[annotation_mode] += 1
                image_shapes[f"{image.shape[0]}x{image.shape[1]}"] += 1
                format_valid += 1
                try:
                    result = extract_tjdr_retinal_fov(
                        image,
                        background_threshold=background_threshold,
                        minimum_area_ratio=minimum_area_ratio,
                        maximum_area_ratio=maximum_area_ratio,
                        substantial_component_ratio=substantial_component_ratio,
                        dominant_component_min_share=dominant_component_min_share,
                    )
                except (RuntimeError, TypeError, ValueError):
                    reason_counts["FOV_EXECUTION_ERROR"] += 1
                    continue
                fov_status_counts[result.status.value] += 1
                fov_method_counts[result.method] += 1
                if result.status is not FOVStatus.VALID or result.mask is None:
                    reason_counts[f"FOV_{result.status.value}"] += 1
                    continue
                if result.area_ratio is None or not math.isfinite(result.area_ratio):
                    reason_counts["FOV_AREA_RATIO_UNAVAILABLE"] += 1
                    continue
                area_ratios.append(result.area_ratio)
                for axis, source_class in TJDR_AXES_TO_CLASSES.items():
                    lesion = np.asarray(annotation == source_class, dtype=np.bool_)
                    outside = np.logical_and(lesion, np.logical_not(result.mask))
                    outside_count = int(np.count_nonzero(outside))
                    if outside_count:
                        outside_fov_image_counts[axis] += 1
                        outside_fov_pixel_counts[axis] += outside_count
                    # Exercise the shared validator while retaining no row value.
                    lesion_outside_fov_ratio(lesion, result.mask)
                decoded += 1
            except TJDRSchemaError as error:
                reason_counts[error.reason_code] += 1
            except KeyError:
                reason_counts["INTERNAL_MEMBER_LOOKUP_ERROR"] += 1
            except (OSError, RuntimeError, TypeError, ValueError):
                reason_counts["UNCLASSIFIED_SAFE_ERROR"] += 1

    zero_outside = not outside_fov_pixel_counts
    return {
        "status": (
            "TJDR_TRAIN_AUDIT_PASS"
            if decoded == expected_count and not reason_counts and zero_outside
            else "TJDR_TRAIN_AUDIT_BLOCKED"
        ),
        "split": "train",
        "expected_pair_count": expected_count,
        "observed_pair_count": len(pairs),
        "format_valid_pair_count": format_valid,
        "eligible_pair_count": decoded,
        "reason_counts": dict(sorted(reason_counts.items())),
        "image_mode_counts": dict(sorted(image_modes.items())),
        "annotation_mode_counts": dict(sorted(annotation_modes.items())),
        "image_shape_counts": dict(sorted(image_shapes.items())),
        "annotation_value_image_counts": {
            str(value): annotation_value_counts[value] for value in sorted(annotation_value_counts)
        },
        "all_background_complete_annotation_count": all_background_count,
        "positive_image_counts": {
            axis: positive_image_counts[axis] for axis in TJDR_AXES_TO_CLASSES
        },
        "fov_status_counts": dict(sorted(fov_status_counts.items())),
        "fov_method_counts": dict(sorted(fov_method_counts.items())),
        "fov_area_ratio_quantiles": _quantiles(area_ratios),
        "lesion_outside_fov_image_counts": {
            axis: outside_fov_image_counts[axis] for axis in TJDR_AXES_TO_CLASSES
        },
        "lesion_outside_fov_pixel_counts": {
            axis: outside_fov_pixel_counts[axis] for axis in TJDR_AXES_TO_CLASSES
        },
        "fov_algorithm_version": TJDR_FOV_ALGORITHM_VERSION,
        "test_scientific_access": False,
        "relations_computed": False,
        "per_instance_rows_persisted": False,
        "source_identifiers_recorded": False,
        "local_paths_recorded": False,
    }


def load_tjdr_train_measurements(
    archive_paths: Sequence[str | Path],
    *,
    expected_count: int,
    background_threshold: float = 5.0,
    minimum_area_ratio: float = 0.10,
    maximum_area_ratio: float = 0.98,
    substantial_component_ratio: float = 0.001,
    dominant_component_min_share: float = 0.90,
    allow_all_background: bool = False,
) -> SourceMeasurements:
    """Measure official train only; any invalid case remains an explicit exclusion."""

    paths = tuple(Path(path) for path in archive_paths)
    records: list[MeasurementRecord] = []
    exclusions = 0
    with ExitStack() as stack:
        archives = tuple(stack.enter_context(zipfile.ZipFile(path)) for path in paths)
        pairs = _index_split(archives, split="train")
        for pair in pairs:
            try:
                image, _ = _read_image(archives, pair.image, rgb=True)
                annotation, _ = _read_image(archives, pair.annotation, rgb=False)
                _validate_annotation(
                    annotation,
                    image.shape[:2],
                    allow_all_background=allow_all_background,
                )
                result = extract_tjdr_retinal_fov(
                    image,
                    background_threshold=background_threshold,
                    minimum_area_ratio=minimum_area_ratio,
                    maximum_area_ratio=maximum_area_ratio,
                    substantial_component_ratio=substantial_component_ratio,
                    dominant_component_min_share=dominant_component_min_share,
                )
                if result.status is not FOVStatus.VALID or result.mask is None:
                    raise ValueError("TJDR FOV is invalid")
                masks = {
                    axis: np.asarray(annotation == source_class, dtype=np.bool_)
                    for axis, source_class in TJDR_AXES_TO_CLASSES.items()
                }
                if any(lesion_outside_fov_ratio(mask, result.mask) > 0 for mask in masks.values()):
                    raise ValueError("TJDR lesion lies outside the FOV")
                records.append(
                    MeasurementRecord(
                        instance_id=opaque_sample_id("tjdr", f"train/{pair.stem}"),
                        primary=_burden(masks, result.mask),
                        variants={},
                        grade=None,
                        grading_split_conflict=False,
                        strict_measurement_quality=True,
                    )
                )
            except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                exclusions += 1
    return SourceMeasurements(
        name="tjdr",
        role="DEVELOPMENT_CALIBRATION",
        expected_input_count=expected_count,
        observed_input_count=len(pairs),
        records=tuple(records),
        measurement_exclusion_count=exclusions,
    )


__all__ = [
    "TJDR_ALLOWED_CLASSES",
    "TJDR_AXES_TO_CLASSES",
    "TJDR_FOV_ALGORITHM_VERSION",
    "TJDRSchemaError",
    "TJDRStructuralBoundary",
    "audit_tjdr_train",
    "extract_tjdr_retinal_fov",
    "inspect_tjdr_structure",
    "load_tjdr_train_measurements",
]

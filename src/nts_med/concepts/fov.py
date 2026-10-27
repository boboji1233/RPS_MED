"""Deterministic field-of-view construction for retinal images.

The extraction functions in this module are independent of lesion masks,
grades, dataset splits, and relation outcomes. Invalid or ambiguous images
fail closed; callers must never substitute the rectangular image area.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral, Real
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

DDR_FOV_ALGORITHM_VERSION: Final = "DDR_BLACK_BACKGROUND_OR_INSCRIBED_ELLIPSE_V2"
DDR_BOUNDING_ELLIPSE_ALGORITHM_VERSION: Final = "DDR_LARGEST_COMPONENT_BOUNDING_ELLIPSE_V1"
DDR_BACKGROUND_THRESHOLD: Final = 5.0
DDR_MIN_AREA_RATIO: Final = 0.10
DDR_MAX_AREA_RATIO: Final = 0.98
DDR_SUBSTANTIAL_COMPONENT_RATIO: Final = 0.001


class FOVStatus(StrEnum):
    """Fail-closed status for one attempted FOV extraction."""

    VALID = "VALID"
    EMPTY = "EMPTY"
    MULTIPLE_COMPONENTS = "MULTIPLE_COMPONENTS"
    AREA_RATIO_OUT_OF_RANGE = "AREA_RATIO_OUT_OF_RANGE"


@dataclass(frozen=True, slots=True)
class FOVResult:
    """Aggregate extraction facts plus a mask only when extraction is valid."""

    status: FOVStatus
    mask: NDArray[np.bool_] | None
    foreground_component_count: int
    substantial_component_count: int
    area_ratio: float | None
    border_contact: bool
    method: str = "THRESHOLDED_LARGEST_COMPONENT"
    algorithm_version: str = DDR_FOV_ALGORITHM_VERSION


def _require_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _validate_rgb_image(image: NDArray[Any]) -> NDArray[np.float64]:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim != 3 or image.shape[2] != 3 or any(size == 0 for size in image.shape):
        raise ValueError("image must be a non-empty HxWx3 array")
    if not (
        np.issubdtype(image.dtype, np.integer)
        or np.issubdtype(image.dtype, np.floating)
    ):
        raise TypeError("image must have a real numeric dtype")
    if not bool(np.isfinite(image).all()):
        raise ValueError("image must not contain NaN or infinity")
    numeric = np.asarray(image, dtype=np.float64)
    if bool((numeric < 0).any()) or bool((numeric > 255).any()):
        raise ValueError("image intensity must be in the closed range [0, 255]")
    return numeric


def _row_runs(row_mask: NDArray[np.bool_]) -> tuple[tuple[int, int], ...]:
    padded = np.pad(row_mask.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return tuple((int(start), int(end)) for start, end in zip(starts, ends, strict=True))


def _component_runs(
    mask: NDArray[np.bool_],
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """Return 8-connected components as compact half-open row runs."""

    parents: list[int] = []
    runs: list[tuple[int, int, int]] = []
    run_ids_by_row: list[list[int]] = []

    def find(item: int) -> int:
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for row in range(mask.shape[0]):
        current_ids: list[int] = []
        for start, end in _row_runs(mask[row]):
            run_id = len(runs)
            runs.append((row, start, end))
            parents.append(run_id)
            current_ids.append(run_id)
            if row:
                for previous_id in run_ids_by_row[row - 1]:
                    _, previous_start, previous_end = runs[previous_id]
                    if start <= previous_end and end >= previous_start:
                        union(run_id, previous_id)
        run_ids_by_row.append(current_ids)

    grouped: defaultdict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for run_id, run in enumerate(runs):
        grouped[find(run_id)].append(run)
    return tuple(tuple(component) for component in grouped.values())


def _fill_holes(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Fill background regions not connected to the image boundary."""

    rows, columns = mask.shape
    background = np.logical_not(mask)
    filled = mask.copy()
    for component in _component_runs(background):
        touches_boundary = any(
            row in (0, rows - 1) or start == 0 or end == columns
            for row, start, end in component
        )
        if not touches_boundary:
            for row, start, end in component:
                filled[row, start:end] = True
    return filled


def _touches_border(mask: NDArray[np.bool_]) -> bool:
    return bool(
        mask[0, :].any() or mask[-1, :].any()
        or mask[:, 0].any() or mask[:, -1].any()
    )


def _validate_binary_mask(mask: NDArray[Any], *, name: str) -> NDArray[np.bool_]:
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if mask.ndim != 2 or any(size == 0 for size in mask.shape):
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not (
        np.issubdtype(mask.dtype, np.bool_)
        or np.issubdtype(mask.dtype, np.integer)
        or np.issubdtype(mask.dtype, np.floating)
    ):
        raise TypeError(f"{name} must have a real numeric or boolean dtype")
    if not bool(np.isfinite(mask).all()):
        raise ValueError(f"{name} must contain only finite values")
    if not bool(np.logical_or(mask == 0, mask == 1).all()):
        raise ValueError(f"{name} must be a canonical 0/1 mask")
    return np.asarray(mask, dtype=np.bool_)


def _require_radius(radius: object) -> int:
    if isinstance(radius, bool) or not isinstance(radius, Integral):
        raise TypeError("radius must be an integer")
    converted = int(radius)
    if converted < 0:
        raise ValueError("radius must be non-negative")
    return converted


def binary_dilate(mask: NDArray[Any], *, radius: int) -> NDArray[np.bool_]:
    """Square-neighbourhood dilation for predeclared mask sensitivities."""

    binary = _validate_binary_mask(mask, name="mask")
    size = _require_radius(radius)
    if size == 0:
        return binary.copy()
    padded = np.pad(binary, size, mode="constant", constant_values=False)
    result = np.zeros(binary.shape, dtype=np.bool_)
    for row_offset in range(2 * size + 1):
        for column_offset in range(2 * size + 1):
            result |= padded[
                row_offset : row_offset + binary.shape[0],
                column_offset : column_offset + binary.shape[1],
            ]
    return result


def binary_erode(mask: NDArray[Any], *, radius: int) -> NDArray[np.bool_]:
    """Square-neighbourhood erosion for predeclared mask/FOV sensitivities."""

    binary = _validate_binary_mask(mask, name="mask")
    size = _require_radius(radius)
    if size == 0:
        return binary.copy()
    padded = np.pad(binary, size, mode="constant", constant_values=False)
    result = np.ones(binary.shape, dtype=np.bool_)
    for row_offset in range(2 * size + 1):
        for column_offset in range(2 * size + 1):
            result &= padded[
                row_offset : row_offset + binary.shape[0],
                column_offset : column_offset + binary.shape[1],
            ]
    return result


def lesion_outside_fov_ratio(
    lesion_mask: NDArray[Any], fov_mask: NDArray[Any]
) -> float:
    """Return the fraction of lesion foreground outside a candidate FOV."""

    lesion = _validate_binary_mask(lesion_mask, name="lesion_mask")
    fov = _validate_binary_mask(fov_mask, name="fov_mask")
    if lesion.shape != fov.shape:
        raise ValueError("lesion_mask and fov_mask must have the same shape")
    lesion_pixels = int(np.count_nonzero(lesion))
    if lesion_pixels == 0:
        return 0.0
    outside = np.logical_and(lesion, np.logical_not(fov))
    return float(np.count_nonzero(outside) / lesion_pixels)


def extract_ddr_retinal_fov(
    image: NDArray[Any],
    *,
    background_threshold: float = DDR_BACKGROUND_THRESHOLD,
    minimum_area_ratio: float = DDR_MIN_AREA_RATIO,
    maximum_area_ratio: float = DDR_MAX_AREA_RATIO,
    substantial_component_ratio: float = DDR_SUBSTANTIAL_COMPONENT_RATIO,
) -> FOVResult:
    """Extract a deterministic DDR retinal FOV from an RGB image.

    Foreground means at least one channel is strictly above the fixed
    black-background threshold. Small disconnected regions are noise. Two or
    more substantial components are ambiguous and fail closed. The largest
    component is hole-filled before its area is checked.
    """

    rgb = _validate_rgb_image(image)
    threshold = _require_finite_real(background_threshold, name="background_threshold")
    minimum = _require_finite_real(minimum_area_ratio, name="minimum_area_ratio")
    maximum = _require_finite_real(maximum_area_ratio, name="maximum_area_ratio")
    substantial_ratio = _require_finite_real(
        substantial_component_ratio, name="substantial_component_ratio"
    )
    if threshold < 0 or threshold >= 255:
        raise ValueError("background_threshold must be in [0, 255)")
    if not 0 < minimum < maximum <= 1:
        raise ValueError("area-ratio bounds must satisfy 0 < minimum < maximum <= 1")
    if not 0 < substantial_ratio <= 1:
        raise ValueError("substantial_component_ratio must be in (0, 1]")

    foreground = np.asarray(np.max(rgb, axis=2) > threshold, dtype=np.bool_)
    components = _component_runs(foreground)
    if not components:
        return FOVResult(FOVStatus.EMPTY, None, 0, 0, None, False)

    image_area = foreground.size
    minimum_substantial_pixels = max(1, math.ceil(image_area * substantial_ratio))
    substantial_count = sum(
        sum(end - start for _, start, end in component) >= minimum_substantial_pixels
        for component in components
    )
    if substantial_count > 1:
        return FOVResult(
            FOVStatus.MULTIPLE_COMPONENTS, None, len(components), substantial_count,
            None, _touches_border(foreground),
        )

    largest = max(
        components,
        key=lambda component: sum(end - start for _, start, end in component),
    )
    largest_mask = np.zeros(foreground.shape, dtype=np.bool_)
    for row, start, end in largest:
        largest_mask[row, start:end] = True
    filled = _fill_holes(largest_mask)
    area_ratio = float(np.count_nonzero(filled) / image_area)
    border_contact = _touches_border(filled)
    if area_ratio > maximum:
        inscribed = official_elliptical_roi_fov(
            (filled.shape[0], filled.shape[1]),
            x0=0,
            y0=0,
            x1=filled.shape[1],
            y1=filled.shape[0],
        )
        return FOVResult(
            FOVStatus.VALID,
            inscribed,
            len(components),
            substantial_count,
            float(np.mean(inscribed)),
            _touches_border(inscribed),
            "FULL_FRAME_INSCRIBED_ELLIPSE",
        )
    if area_ratio < minimum:
        return FOVResult(
            FOVStatus.AREA_RATIO_OUT_OF_RANGE, None, len(components),
            substantial_count, area_ratio, border_contact,
        )
    return FOVResult(
        FOVStatus.VALID, filled, len(components), substantial_count,
        area_ratio, border_contact,
    )


def extract_ddr_bounding_ellipse_fov(
    image: NDArray[Any],
    *,
    background_threshold: float = DDR_BACKGROUND_THRESHOLD,
    minimum_area_ratio: float = DDR_MIN_AREA_RATIO,
    maximum_area_ratio: float = 1.0,
    substantial_component_ratio: float = DDR_SUBSTANTIAL_COMPONENT_RATIO,
) -> FOVResult:
    """Construct a deterministic bounding ellipse around the largest component.

    This is a sensitivity implementation only.  Multiple substantial retinal
    candidates still fail closed, and the rectangular image area is never used
    as the FOV denominator.
    """

    rgb = _validate_rgb_image(image)
    threshold = _require_finite_real(background_threshold, name="background_threshold")
    minimum = _require_finite_real(minimum_area_ratio, name="minimum_area_ratio")
    maximum = _require_finite_real(maximum_area_ratio, name="maximum_area_ratio")
    substantial_ratio = _require_finite_real(
        substantial_component_ratio, name="substantial_component_ratio"
    )
    if threshold < 0 or threshold >= 255:
        raise ValueError("background_threshold must be in [0, 255)")
    if not 0 < minimum < maximum <= 1:
        raise ValueError("area-ratio bounds must satisfy 0 < minimum < maximum <= 1")
    if not 0 < substantial_ratio <= 1:
        raise ValueError("substantial_component_ratio must be in (0, 1]")
    foreground = np.asarray(np.max(rgb, axis=2) > threshold, dtype=np.bool_)
    components = _component_runs(foreground)
    if not components:
        return FOVResult(
            FOVStatus.EMPTY, None, 0, 0, None, False,
            "LARGEST_COMPONENT_BOUNDING_ELLIPSE", DDR_BOUNDING_ELLIPSE_ALGORITHM_VERSION,
        )
    image_area = foreground.size
    minimum_substantial_pixels = max(1, math.ceil(image_area * substantial_ratio))
    substantial_count = sum(
        sum(end - start for _, start, end in component) >= minimum_substantial_pixels
        for component in components
    )
    if substantial_count > 1:
        return FOVResult(
            FOVStatus.MULTIPLE_COMPONENTS, None, len(components), substantial_count,
            None, _touches_border(foreground), "LARGEST_COMPONENT_BOUNDING_ELLIPSE",
            DDR_BOUNDING_ELLIPSE_ALGORITHM_VERSION,
        )
    largest = max(
        components,
        key=lambda component: sum(end - start for _, start, end in component),
    )
    top = min(row for row, _, _ in largest)
    bottom = max(row for row, _, _ in largest) + 1
    left = min(start for _, start, _ in largest)
    right = max(end for _, _, end in largest)
    ellipse = official_elliptical_roi_fov(
        (foreground.shape[0], foreground.shape[1]),
        x0=left,
        y0=top,
        x1=right,
        y1=bottom,
    )
    area_ratio = float(np.mean(ellipse))
    if area_ratio < minimum or area_ratio > maximum:
        return FOVResult(
            FOVStatus.AREA_RATIO_OUT_OF_RANGE, None, len(components), substantial_count,
            area_ratio, _touches_border(ellipse), "LARGEST_COMPONENT_BOUNDING_ELLIPSE",
            DDR_BOUNDING_ELLIPSE_ALGORITHM_VERSION,
        )
    return FOVResult(
        FOVStatus.VALID, ellipse, len(components), substantial_count, area_ratio,
        _touches_border(ellipse), "LARGEST_COMPONENT_BOUNDING_ELLIPSE",
        DDR_BOUNDING_ELLIPSE_ALGORITHM_VERSION,
    )


def official_elliptical_roi_fov(
    image_shape: tuple[int, int],
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> NDArray[np.bool_]:
    """Reconstruct an ellipse inside an official half-open ROI bounding box."""

    if (
        len(image_shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, Integral) for value in image_shape)
    ):
        raise TypeError("image_shape must contain two integers")
    height, width = (int(value) for value in image_shape)
    if height <= 0 or width <= 0:
        raise ValueError("image_shape dimensions must be positive")
    bounds = (x0, y0, x1, y1)
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in bounds):
        raise TypeError("ROI bounds must be integers")
    left, top, right, bottom = (int(value) for value in bounds)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("ROI bounds must form a non-empty box inside the image")

    x_radius = (right - left) / 2.0
    y_radius = (bottom - top) / 2.0
    x_center = left + x_radius
    y_center = top + y_radius
    rows, columns = np.ogrid[:height, :width]
    normalized = (
        ((columns + 0.5 - x_center) / x_radius) ** 2
        + ((rows + 0.5 - y_center) / y_radius) ** 2
    )
    return np.asarray(normalized <= 1.0, dtype=np.bool_)


__all__ = [
    "DDR_BACKGROUND_THRESHOLD", "DDR_BOUNDING_ELLIPSE_ALGORITHM_VERSION",
    "DDR_FOV_ALGORITHM_VERSION",
    "DDR_MAX_AREA_RATIO", "DDR_MIN_AREA_RATIO",
    "DDR_SUBSTANTIAL_COMPONENT_RATIO", "FOVResult", "FOVStatus",
    "binary_dilate", "binary_erode", "extract_ddr_bounding_ellipse_fov",
    "extract_ddr_retinal_fov", "lesion_outside_fov_ratio",
    "official_elliptical_roi_fov",
]

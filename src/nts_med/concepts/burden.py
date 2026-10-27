"""Pure NumPy lesion-burden calculations.

The functions in this module deliberately do not guess how a label mask should
be binarized.  Canonical binary masks contain only zero and one.  A label mask
must instead be accompanied by an explicit ``positive_class`` value.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral, Real
from typing import Any, Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

Numeric: TypeAlias = int | float | np.integer[Any] | np.floating[Any]

DR_LESION_AXES: Final[tuple[str, ...]] = ("ma", "hem", "ex", "cws")
"""The four preregistered primary diabetic-retinopathy lesion axes."""

DEFAULT_DR_SCALE: Final[Mapping[str, float]] = {
    "ma": 1.0,
    "hem": 1.0,
    "ex": 1.0,
    "cws": 1.0,
}
"""Explicit, neutral M0--M1 scaling defaults.

These values are engineering defaults, not calibrated clinical thresholds.
Callers should persist the scale used with every generated burden artifact.
"""


@dataclass(frozen=True, slots=True)
class BurdenMetrics:
    """Untransformed burden measurements for one lesion mask."""

    fov_area: int
    component_count: int
    component_count_density: float
    lesion_area_pixels: int
    lesion_area_ratio: float


@dataclass(frozen=True, slots=True)
class BurdenUnitChain:
    """Explicit units from raw foreground evidence to relation thresholds."""

    axis: str
    raw_measure: str
    fov_denominator_pixels: int
    normalized_measure: str
    density_scale: float
    transformed_measure: str
    relation_threshold_unit: str
    delta: float
    epsilon: float


class LesionMaskStatus(StrEnum):
    """Evidence states kept distinct before burden calculation."""

    MISSING = "MISSING"
    PRESENT_EMPTY = "PRESENT_EMPTY"
    PRESENT_POSITIVE = "PRESENT_POSITIVE"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class LesionMaskInspection:
    """Path-free validation result for one in-memory lesion mask."""

    status: LesionMaskStatus
    mask: NDArray[np.bool_] | None


def _require_real_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _require_min_component_pixels(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("min_component_pixels must be an integer")
    converted = int(value)
    if converted < 0:
        raise ValueError("min_component_pixels must be non-negative")
    return converted


def burden_unit_chain(
    axis: str,
    *,
    fov_area: int,
    density_scale: Real,
    delta: Real,
    epsilon: Real,
) -> BurdenUnitChain:
    """Validate and record the complete numerical unit chain for one axis."""

    if axis not in DR_LESION_AXES:
        raise ValueError(f"axis must be one of {DR_LESION_AXES}")
    if isinstance(fov_area, bool) or not isinstance(fov_area, Integral):
        raise TypeError("fov_area must be an integer")
    if int(fov_area) <= 0:
        raise ValueError("fov_area must be positive")
    scale_value = _require_real_scalar(density_scale, name="density_scale")
    delta_value = _require_real_scalar(delta, name="delta")
    epsilon_value = _require_real_scalar(epsilon, name="epsilon")
    if scale_value <= 0:
        raise ValueError("density_scale must be positive")
    if delta_value < 0 or epsilon_value < 0:
        raise ValueError("relation thresholds must be non-negative")
    raw_measure = "CONNECTED_COMPONENT_COUNT" if axis == "ma" else "LESION_FOREGROUND_PIXELS"
    normalized_measure = (
        "COMPONENTS_PER_FOV_PIXEL" if axis == "ma" else "LESION_PIXELS_PER_FOV_PIXEL"
    )
    return BurdenUnitChain(
        axis=axis,
        raw_measure=raw_measure,
        fov_denominator_pixels=int(fov_area),
        normalized_measure=normalized_measure,
        density_scale=scale_value,
        transformed_measure="LOG1P_NORMALIZED_MEASURE_TIMES_DENSITY_SCALE",
        relation_threshold_unit="LOG1P_SCALED_FOV_NORMALIZED_BURDEN",
        delta=delta_value,
        epsilon=epsilon_value,
    )


def _require_connectivity(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("connectivity must be an integer")
    converted = int(value)
    if converted not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    return converted


def validate_binary_mask(
    mask: NDArray[Any],
    *,
    positive_class: Numeric | None = None,
    name: str = "mask",
) -> NDArray[np.bool_]:
    """Validate and return a two-dimensional boolean mask.

    Boolean and numeric arrays containing only 0/1 are accepted directly.
    Any other numeric label representation requires ``positive_class``.  This
    includes a two-valued mask such as ``{0, 255}``; it is not silently treated
    as binary.
    """

    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if mask.ndim != 2 or any(dimension == 0 for dimension in mask.shape):
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not (
        np.issubdtype(mask.dtype, np.bool_)
        or np.issubdtype(mask.dtype, np.integer)
        or np.issubdtype(mask.dtype, np.floating)
    ):
        raise TypeError(f"{name} must have a real numeric or boolean dtype")
    if not bool(np.isfinite(mask).all()):
        raise ValueError(f"{name} must not contain NaN or infinity")

    if positive_class is not None:
        class_value = _require_real_scalar(positive_class, name="positive_class")
        return np.asarray(mask == class_value, dtype=np.bool_)

    canonical_binary = np.logical_or(mask == 0, mask == 1)
    if not bool(canonical_binary.all()):
        raise ValueError(f"{name} is not a canonical 0/1 mask; pass positive_class explicitly")
    return np.asarray(mask, dtype=np.bool_)


def inspect_lesion_mask(
    mask: NDArray[Any] | None,
    *,
    positive_class: Numeric | None = None,
) -> LesionMaskInspection:
    """Keep missing, empty, positive, and invalid mask evidence distinct."""

    if mask is None:
        return LesionMaskInspection(LesionMaskStatus.MISSING, None)
    try:
        binary = validate_binary_mask(
            mask, positive_class=positive_class, name="lesion_mask"
        )
    except (TypeError, ValueError):
        return LesionMaskInspection(LesionMaskStatus.INVALID, None)
    status = (
        LesionMaskStatus.PRESENT_POSITIVE
        if bool(binary.any())
        else LesionMaskStatus.PRESENT_EMPTY
    )
    return LesionMaskInspection(status, binary)


def compute_fov_area(fov_mask: NDArray[Any]) -> int:
    """Return the number of pixels inside a validated field-of-view mask."""

    fov = validate_binary_mask(fov_mask, name="fov_mask")
    return int(np.count_nonzero(fov))


def _component_sizes(mask: NDArray[np.bool_], *, connectivity: int) -> list[int]:
    validated_connectivity = _require_connectivity(connectivity)
    neighbours: tuple[tuple[int, int], ...]

    if validated_connectivity == 4:
        neighbours = ((-1, 0), (0, -1), (0, 1), (1, 0))
    else:
        neighbours = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )

    rows, columns = mask.shape
    visited = np.zeros(mask.shape, dtype=np.bool_)
    sizes: list[int] = []
    for row, column in np.argwhere(mask):
        row_index = int(row)
        column_index = int(column)
        if visited[row_index, column_index]:
            continue
        visited[row_index, column_index] = True
        queue: deque[tuple[int, int]] = deque([(row_index, column_index)])
        size = 0
        while queue:
            current_row, current_column = queue.pop()
            size += 1
            for row_delta, column_delta in neighbours:
                next_row = current_row + row_delta
                next_column = current_column + column_delta
                if (
                    0 <= next_row < rows
                    and 0 <= next_column < columns
                    and mask[next_row, next_column]
                    and not visited[next_row, next_column]
                ):
                    visited[next_row, next_column] = True
                    queue.append((next_row, next_column))
        sizes.append(size)
    return sizes


def connected_component_count(
    lesion_mask: NDArray[Any],
    *,
    min_component_pixels: int = 1,
    connectivity: int = 8,
    positive_class: Numeric | None = None,
) -> int:
    """Count connected components meeting the inclusive size threshold."""

    minimum = _require_min_component_pixels(min_component_pixels)
    binary = validate_binary_mask(lesion_mask, positive_class=positive_class, name="lesion_mask")
    return sum(size >= minimum for size in _component_sizes(binary, connectivity=connectivity))


def component_count_density(component_count: int, fov_area: int) -> float:
    """Return components per FOV pixel with explicit zero-area validation."""

    if isinstance(component_count, (bool, np.bool_)) or not isinstance(component_count, Integral):
        raise TypeError("component_count must be an integer")
    if isinstance(fov_area, (bool, np.bool_)) or not isinstance(fov_area, Integral):
        raise TypeError("fov_area must be an integer")
    count = int(component_count)
    area = int(fov_area)
    if count < 0:
        raise ValueError("component_count must be non-negative")
    if area <= 0:
        raise ValueError("fov_area must be positive")
    return count / area


def safe_log1p(value: Numeric, *, scale: Numeric = 1.0) -> float:
    """Apply ``log1p(value * scale)`` after finite, non-negative validation."""

    numeric_value = _require_real_scalar(value, name="value")
    numeric_scale = _require_real_scalar(scale, name="scale")
    if numeric_value < 0:
        raise ValueError("value must be non-negative")
    if numeric_scale < 0:
        raise ValueError("scale must be non-negative")
    product = numeric_value * numeric_scale
    if not math.isfinite(product):
        raise ValueError("value * scale must be finite")
    return math.log1p(product)


def compute_lesion_metrics(
    lesion_mask: NDArray[Any],
    fov_mask: NDArray[Any],
    *,
    min_component_pixels: int = 1,
    connectivity: int = 8,
    positive_class: Numeric | None = None,
) -> BurdenMetrics:
    """Compute count and area burden for one lesion inside the FOV.

    Components smaller than ``min_component_pixels`` are excluded from both
    the count and lesion area.  Lesion pixels outside the FOV are ignored.
    """

    minimum = _require_min_component_pixels(min_component_pixels)
    fov = validate_binary_mask(fov_mask, name="fov_mask")
    lesion = validate_binary_mask(lesion_mask, positive_class=positive_class, name="lesion_mask")
    if lesion.shape != fov.shape:
        raise ValueError("lesion_mask and fov_mask must have the same shape")

    fov_area = int(np.count_nonzero(fov))
    if fov_area == 0:
        raise ValueError("fov_mask must contain at least one foreground pixel")
    lesion_in_fov = np.logical_and(lesion, fov)
    retained_sizes = [
        size
        for size in _component_sizes(lesion_in_fov, connectivity=connectivity)
        if size >= minimum
    ]
    count = len(retained_sizes)
    lesion_pixels = sum(retained_sizes)
    return BurdenMetrics(
        fov_area=fov_area,
        component_count=count,
        component_count_density=component_count_density(count, fov_area),
        lesion_area_pixels=lesion_pixels,
        lesion_area_ratio=lesion_pixels / fov_area,
    )


def lesion_area_ratio(
    lesion_mask: NDArray[Any],
    fov_mask: NDArray[Any],
    *,
    min_component_pixels: int = 1,
    connectivity: int = 8,
    positive_class: Numeric | None = None,
) -> float:
    """Return the retained lesion-pixel fraction inside the FOV."""

    return compute_lesion_metrics(
        lesion_mask,
        fov_mask,
        min_component_pixels=min_component_pixels,
        connectivity=connectivity,
        positive_class=positive_class,
    ).lesion_area_ratio


def _validated_axis_mapping(values: Mapping[str, object], *, name: str) -> None:
    actual = set(values)
    expected = set(DR_LESION_AXES)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} axes mismatch; missing={missing}, extra={extra}")


def _axis_scales(scale: Numeric | Mapping[str, Numeric]) -> dict[str, float]:
    if isinstance(scale, Mapping):
        _validated_axis_mapping(scale, name="scale")
        return {
            axis: _require_real_scalar(scale[axis], name=f"scale[{axis!r}]")
            for axis in DR_LESION_AXES
        }
    scalar = _require_real_scalar(scale, name="scale")
    return {axis: scalar for axis in DR_LESION_AXES}


def _axis_positive_classes(
    positive_class: Numeric | Mapping[str, Numeric | None] | None,
) -> dict[str, Numeric | None]:
    if isinstance(positive_class, Mapping):
        _validated_axis_mapping(positive_class, name="positive_class")
        return {axis: positive_class[axis] for axis in DR_LESION_AXES}
    return {axis: positive_class for axis in DR_LESION_AXES}


def compute_dr_burden(
    lesion_masks: Mapping[str, NDArray[Any]],
    fov_mask: NDArray[Any],
    min_component_pixels: Mapping[str, int],
    *,
    scale: Numeric | Mapping[str, Numeric] = DEFAULT_DR_SCALE,
    connectivity: int = 8,
    positive_class: Numeric | Mapping[str, Numeric | None] | None = None,
) -> dict[str, float]:
    """Compute the preregistered four-axis DR burden representation.

    ``ma`` is the scaled log component-count density.  ``hem``, ``ex`` and
    ``cws`` are scaled log lesion-area ratios.  The returned mapping follows
    the fixed ``DR_LESION_AXES`` order.
    """

    _validated_axis_mapping(lesion_masks, name="lesion_masks")
    _validated_axis_mapping(min_component_pixels, name="min_component_pixels")
    scales = _axis_scales(scale)
    positive_classes = _axis_positive_classes(positive_class)
    if any(value < 0 for value in scales.values()):
        raise ValueError("all scale values must be non-negative")

    fov = validate_binary_mask(fov_mask, name="fov_mask")
    fov_area = int(np.count_nonzero(fov))
    if fov_area == 0:
        raise ValueError("fov_mask must contain at least one foreground pixel")
    burdens: dict[str, float] = {}
    for axis in DR_LESION_AXES:
        minimum = _require_min_component_pixels(min_component_pixels[axis])
        if axis != "ma" and minimum == 1:
            lesion = validate_binary_mask(
                lesion_masks[axis],
                positive_class=positive_classes[axis],
                name="lesion_mask",
            )
            if lesion.shape != fov.shape:
                raise ValueError("lesion_mask and fov_mask must have the same shape")
            raw_value = float(np.count_nonzero(np.logical_and(lesion, fov)) / fov_area)
            burdens[axis] = safe_log1p(raw_value, scale=scales[axis])
            continue
        metrics = compute_lesion_metrics(
            lesion_masks[axis],
            fov,
            min_component_pixels=minimum,
            connectivity=connectivity,
            positive_class=positive_classes[axis],
        )
        raw_value = metrics.component_count_density if axis == "ma" else metrics.lesion_area_ratio
        burdens[axis] = safe_log1p(raw_value, scale=scales[axis])
    return burdens

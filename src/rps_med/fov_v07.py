"""Versioned dominant-component FOV amendment for RPS-Med R1 v0.7.

The v0.6 extractor remains immutable.  This module changes behavior only when
that extractor reports multiple substantial foreground components.  It never
uses lesion masks, grades, split outcomes, or relation results.
"""

from __future__ import annotations

import math
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from nts_med.concepts import fov as fov_core
from nts_med.concepts.fov import FOVResult, FOVStatus

DDR_FOV_V07_ALGORITHM_VERSION: Final = "DDR_DOMINANT_COMPONENT_V1"
DDR_DOMINANT_COMPONENT_MIN_SHARE: Final = 0.90


def _dominance_threshold(value: object) -> float:
    threshold = fov_core._require_finite_real(value, name="dominant_component_min_share")
    if not 0.5 < threshold <= 1.0:
        raise ValueError("dominant_component_min_share must be in (0.5, 1]")
    return threshold


def extract_ddr_retinal_fov_v07(
    image: NDArray[Any],
    *,
    background_threshold: float = fov_core.DDR_BACKGROUND_THRESHOLD,
    minimum_area_ratio: float = fov_core.DDR_MIN_AREA_RATIO,
    maximum_area_ratio: float = fov_core.DDR_MAX_AREA_RATIO,
    substantial_component_ratio: float = fov_core.DDR_SUBSTANTIAL_COMPONENT_RATIO,
    dominant_component_min_share: float = DDR_DOMINANT_COMPONENT_MIN_SHARE,
) -> FOVResult:
    """Resolve only clearly dominant multi-component retinal foregrounds.

    The legacy result is returned byte-for-byte at the array level whenever it
    is not ``MULTIPLE_COMPONENTS``.  For multiple substantial components, the
    uniquely largest component must hold at least the preregistered fraction of
    all substantial foreground pixels.  Otherwise ambiguity remains fail-closed.
    """

    legacy = fov_core.extract_ddr_retinal_fov(
        image,
        background_threshold=background_threshold,
        minimum_area_ratio=minimum_area_ratio,
        maximum_area_ratio=maximum_area_ratio,
        substantial_component_ratio=substantial_component_ratio,
    )
    dominance = _dominance_threshold(dominant_component_min_share)
    if legacy.status is not FOVStatus.MULTIPLE_COMPONENTS:
        return legacy

    rgb = fov_core._validate_rgb_image(image)
    threshold = float(background_threshold)
    foreground = np.asarray(np.max(rgb, axis=2) > threshold, dtype=np.bool_)
    components = fov_core._component_runs(foreground)
    image_area = foreground.size
    minimum_substantial_pixels = max(
        1, math.ceil(image_area * float(substantial_component_ratio))
    )
    sized = tuple(
        (
            sum(end - start for _, start, end in component),
            component,
        )
        for component in components
    )
    substantial = tuple(item for item in sized if item[0] >= minimum_substantial_pixels)
    if len(substantial) < 2:
        raise RuntimeError("legacy multiple-component status is internally inconsistent")

    ordered = sorted(substantial, key=lambda item: item[0], reverse=True)
    largest_size, largest = ordered[0]
    total_substantial = sum(size for size, _ in ordered)
    largest_share = largest_size / total_substantial
    unique_largest = largest_size > ordered[1][0]
    if not unique_largest or largest_share < dominance:
        return FOVResult(
            FOVStatus.MULTIPLE_COMPONENTS,
            None,
            len(components),
            len(substantial),
            None,
            fov_core._touches_border(foreground),
            "AMBIGUOUS_MULTIPLE_SUBSTANTIAL_COMPONENTS",
            DDR_FOV_V07_ALGORITHM_VERSION,
        )

    largest_mask = np.zeros(foreground.shape, dtype=np.bool_)
    for row, start, end in largest:
        largest_mask[row, start:end] = True
    filled = fov_core._fill_holes(largest_mask)
    area_ratio = float(np.count_nonzero(filled) / image_area)
    border_contact = fov_core._touches_border(filled)
    if area_ratio > float(maximum_area_ratio):
        inscribed = fov_core.official_elliptical_roi_fov(
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
            len(substantial),
            float(np.mean(inscribed)),
            fov_core._touches_border(inscribed),
            "DOMINANT_COMPONENT_FULL_FRAME_INSCRIBED_ELLIPSE",
            DDR_FOV_V07_ALGORITHM_VERSION,
        )
    if area_ratio < float(minimum_area_ratio):
        return FOVResult(
            FOVStatus.AREA_RATIO_OUT_OF_RANGE,
            None,
            len(components),
            len(substantial),
            area_ratio,
            border_contact,
            "THRESHOLDED_DOMINANT_COMPONENT",
            DDR_FOV_V07_ALGORITHM_VERSION,
        )
    return FOVResult(
        FOVStatus.VALID,
        filled,
        len(components),
        len(substantial),
        area_ratio,
        border_contact,
        "THRESHOLDED_DOMINANT_COMPONENT",
        DDR_FOV_V07_ALGORITHM_VERSION,
    )


__all__ = [
    "DDR_DOMINANT_COMPONENT_MIN_SHARE",
    "DDR_FOV_V07_ALGORITHM_VERSION",
    "extract_ddr_retinal_fov_v07",
]

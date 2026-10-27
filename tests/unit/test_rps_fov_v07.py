from __future__ import annotations

import numpy as np
import pytest

from nts_med.concepts.fov import FOVStatus, extract_ddr_retinal_fov
from rps_med.fov_v07 import (
    DDR_FOV_V07_ALGORITHM_VERSION,
    extract_ddr_retinal_fov_v07,
)


def _image_with_components(
    components: tuple[tuple[slice, slice], ...], *, shape: tuple[int, int] = (100, 100)
) -> np.ndarray:
    image = np.zeros((*shape, 3), dtype=np.uint8)
    for rows, columns in components:
        image[rows, columns] = 80
    return image


def test_legacy_valid_fov_is_pixelwise_preserved() -> None:
    image = _image_with_components(((slice(10, 90), slice(10, 90)),))
    legacy = extract_ddr_retinal_fov(image)
    amended = extract_ddr_retinal_fov_v07(image)

    assert legacy.status is FOVStatus.VALID
    assert amended.status is FOVStatus.VALID
    assert legacy.mask is not None and amended.mask is not None
    assert np.array_equal(legacy.mask, amended.mask)
    assert legacy.method == amended.method
    assert legacy.algorithm_version == amended.algorithm_version


def test_dominant_component_resolves_small_but_substantial_artifact() -> None:
    image = _image_with_components(
        ((slice(10, 90), slice(10, 90)), (slice(0, 5), slice(0, 5)))
    )
    legacy = extract_ddr_retinal_fov(image)
    amended = extract_ddr_retinal_fov_v07(image)

    assert legacy.status is FOVStatus.MULTIPLE_COMPONENTS
    assert amended.status is FOVStatus.VALID
    assert amended.mask is not None
    assert amended.method == "THRESHOLDED_DOMINANT_COMPONENT"
    assert amended.algorithm_version == DDR_FOV_V07_ALGORITHM_VERSION
    assert np.count_nonzero(amended.mask) == 80 * 80


def test_equal_or_non_dominant_components_remain_fail_closed() -> None:
    equal = _image_with_components(
        ((slice(5, 35), slice(5, 35)), (slice(60, 90), slice(60, 90)))
    )
    below = _image_with_components(
        ((slice(5, 65), slice(5, 65)), (slice(75, 96), slice(70, 90)))
    )

    for image in (equal, below):
        result = extract_ddr_retinal_fov_v07(image)
        assert result.status is FOVStatus.MULTIPLE_COMPONENTS
        assert result.mask is None
        assert result.algorithm_version == DDR_FOV_V07_ALGORITHM_VERSION


def test_dominance_boundary_is_inclusive_and_deterministic() -> None:
    image = _image_with_components(
        ((slice(5, 65), slice(5, 65)), (slice(75, 95), slice(70, 90)))
    )
    first = extract_ddr_retinal_fov_v07(image, dominant_component_min_share=0.90)
    second = extract_ddr_retinal_fov_v07(image, dominant_component_min_share=0.90)

    assert first.status is FOVStatus.VALID
    assert first.mask is not None and second.mask is not None
    assert np.array_equal(first.mask, second.mask)
    assert first.area_ratio == second.area_ratio


def test_invalid_dominance_threshold_fails_explicitly() -> None:
    image = _image_with_components(((slice(10, 90), slice(10, 90)),))
    with pytest.raises(ValueError, match="dominant_component_min_share"):
        extract_ddr_retinal_fov_v07(image, dominant_component_min_share=0.5)

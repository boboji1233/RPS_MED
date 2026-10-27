"""Development-only calibration of source-specific FOV-ratio cutoffs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np


@dataclass(frozen=True, slots=True)
class FOVRatioCutoffs:
    """Inclusive source-specific bounds for the extreme-FOV sensitivity."""

    lower: float
    upper: float
    lower_quantile: float
    upper_quantile: float
    quantile_method: str
    instance_count: int


def calibrate_fov_ratio_cutoffs(
    ratios: Sequence[Real],
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> FOVRatioCutoffs:
    """Return inclusive Q01/Q99 bounds using NumPy's ``higher`` method."""

    if isinstance(ratios, (str, bytes)) or not ratios:
        raise ValueError("ratios must be a non-empty numeric sequence")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    values: list[float] = []
    for ratio in ratios:
        if isinstance(ratio, bool) or not isinstance(ratio, Real):
            raise TypeError("FOV ratios must be real numbers")
        numeric = float(ratio)
        if not math.isfinite(numeric) or not 0.0 < numeric <= 1.0:
            raise ValueError("FOV ratios must be finite and in (0, 1]")
        values.append(numeric)
    lower, upper = np.quantile(
        np.asarray(values, dtype=np.float64),
        [lower_quantile, upper_quantile],
        method="higher",
    )
    return FOVRatioCutoffs(
        lower=float(lower),
        upper=float(upper),
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        quantile_method="HIGHER",
        instance_count=len(values),
    )


def fov_ratio_is_within_cutoffs(ratio: Real, cutoffs: FOVRatioCutoffs) -> bool:
    """Return whether a ratio lies inside the inclusive frozen interval."""

    if isinstance(ratio, bool) or not isinstance(ratio, Real):
        raise TypeError("FOV ratio must be a real number")
    numeric = float(ratio)
    if not math.isfinite(numeric) or not 0.0 < numeric <= 1.0:
        raise ValueError("FOV ratio must be finite and in (0, 1]")
    return cutoffs.lower <= numeric <= cutoffs.upper


__all__ = [
    "FOVRatioCutoffs",
    "calibrate_fov_ratio_cutoffs",
    "fov_ratio_is_within_cutoffs",
]

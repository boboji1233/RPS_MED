"""Outcome-independent threshold-transfer calibration on development splits.

The functions in this module operate on already computed, path-free burden
values.  They never infer relations and never persist instance pairs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class TransferEstimate:
    """One transferred ``delta``/``epsilon`` estimate with uncertainty."""

    delta: float
    epsilon: float
    delta_ci: tuple[float, float]
    epsilon_ci: tuple[float, float]
    valid_bootstrap_replicates: int


@dataclass(frozen=True, slots=True)
class QuantileMatchedAxis:
    """Empirical-CDF threshold transfer for one lesion axis."""

    estimate: TransferEstimate
    source_instance_count: int
    target_instance_count: int
    source_pair_count: int
    target_pair_count: int
    delta_source_percentile: float
    epsilon_source_percentile: float


@dataclass(frozen=True, slots=True)
class MorphologyNoiseAxis:
    """Threshold estimate from per-image morphology perturbation magnitudes."""

    estimate: TransferEstimate
    instance_count: int


def _validated_values(values: Sequence[Real], *, name: str, minimum: int) -> NDArray[np.float64]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of real numbers")
    converted: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must contain real numbers")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"{name} must contain finite non-negative values")
        converted.append(numeric)
    if len(converted) < minimum:
        raise ValueError(f"{name} requires at least {minimum} values")
    return np.asarray(converted, dtype=np.float64)


def _validated_primary(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


def _validated_bootstrap(
    replicates: object,
    seed: object,
    confidence_level: object,
) -> tuple[int, int, float]:
    if isinstance(replicates, bool) or not isinstance(replicates, Integral):
        raise TypeError("bootstrap_replicates must be an integer")
    count = int(replicates)
    if count < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    if isinstance(confidence_level, bool) or not isinstance(confidence_level, Real):
        raise TypeError("confidence_level must be a real number")
    confidence = float(confidence_level)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be finite and in (0, 1)")
    return count, int(seed), confidence


def _pair_geometry(values: NDArray[np.float64]) -> tuple[
    NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]
]:
    left, right = np.triu_indices(len(values), k=1)
    differences = np.abs(values[left] - values[right])
    order = np.argsort(differences, kind="stable")
    return differences[order], left[order], right[order]


def _right_ecdf(sorted_values: NDArray[np.float64], threshold: float) -> float:
    return float(np.searchsorted(sorted_values, threshold, side="right") / len(sorted_values))


def _higher_empirical_quantile(sorted_values: NDArray[np.float64], probability: float) -> float:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    rank = max(1, math.ceil(probability * len(sorted_values)))
    return float(sorted_values[min(rank - 1, len(sorted_values) - 1)])


def _weighted_right_ecdf(
    sorted_values: NDArray[np.float64],
    weights: NDArray[np.int64],
    threshold: float,
) -> float:
    total = int(np.sum(weights, dtype=np.int64))
    if total <= 0:
        raise ValueError("bootstrap resample contains no legal distinct-instance pairs")
    boundary = int(np.searchsorted(sorted_values, threshold, side="right"))
    return float(np.sum(weights[:boundary], dtype=np.int64) / total)


def _weighted_higher_quantile(
    sorted_values: NDArray[np.float64],
    weights: NDArray[np.int64],
    probability: float,
) -> float:
    total = int(np.sum(weights, dtype=np.int64))
    if total <= 0:
        raise ValueError("bootstrap resample contains no legal distinct-instance pairs")
    required = max(1, math.ceil(probability * total))
    index = int(np.searchsorted(np.cumsum(weights, dtype=np.int64), required, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _interval(values: Sequence[float], confidence_level: float) -> tuple[float, float]:
    alpha = (1.0 - confidence_level) / 2.0
    return (
        float(np.quantile(values, alpha, method="linear")),
        float(np.quantile(values, 1.0 - alpha, method="linear")),
    )


def calibrate_quantile_matched_axis(
    source_burdens: Sequence[Real],
    target_burdens: Sequence[Real],
    *,
    primary_delta: Real,
    primary_epsilon: Real,
    bootstrap_replicates: int = 5000,
    seed: int = 20260823,
    confidence_level: float = 0.95,
) -> QuantileMatchedAxis:
    """Transfer source thresholds by their empirical pair-difference percentile.

    Bootstrap replicates resample image instances.  Pair rows are never sampled;
    multiplicity weights are applied only to legal pairs of distinct source
    instances.
    """

    source = _validated_values(source_burdens, name="source_burdens", minimum=2)
    target = _validated_values(target_burdens, name="target_burdens", minimum=2)
    delta = _validated_primary(primary_delta, name="primary_delta")
    epsilon = _validated_primary(primary_epsilon, name="primary_epsilon")
    replicate_count, resolved_seed, confidence = _validated_bootstrap(
        bootstrap_replicates, seed, confidence_level
    )
    source_differences, source_left, source_right = _pair_geometry(source)
    target_differences, target_left, target_right = _pair_geometry(target)
    delta_percentile = _right_ecdf(source_differences, delta)
    epsilon_percentile = _right_ecdf(source_differences, epsilon)
    point_delta = _higher_empirical_quantile(target_differences, delta_percentile)
    point_epsilon = _higher_empirical_quantile(target_differences, epsilon_percentile)

    rng = np.random.default_rng(resolved_seed)
    bootstrap_delta: list[float] = []
    bootstrap_epsilon: list[float] = []
    for _ in range(replicate_count):
        source_counts = np.bincount(
            rng.integers(0, len(source), size=len(source)), minlength=len(source)
        ).astype(np.int64, copy=False)
        target_counts = np.bincount(
            rng.integers(0, len(target), size=len(target)), minlength=len(target)
        ).astype(np.int64, copy=False)
        source_weights = source_counts[source_left] * source_counts[source_right]
        target_weights = target_counts[target_left] * target_counts[target_right]
        if int(source_weights.sum()) == 0 or int(target_weights.sum()) == 0:
            continue
        replicate_delta_percentile = _weighted_right_ecdf(
            source_differences, source_weights, delta
        )
        replicate_epsilon_percentile = _weighted_right_ecdf(
            source_differences, source_weights, epsilon
        )
        bootstrap_delta.append(
            _weighted_higher_quantile(
                target_differences, target_weights, replicate_delta_percentile
            )
        )
        bootstrap_epsilon.append(
            _weighted_higher_quantile(
                target_differences, target_weights, replicate_epsilon_percentile
            )
        )
    if not bootstrap_delta:
        raise ValueError("all quantile-matched bootstrap replicates were degenerate")
    return QuantileMatchedAxis(
        estimate=TransferEstimate(
            delta=point_delta,
            epsilon=point_epsilon,
            delta_ci=_interval(bootstrap_delta, confidence),
            epsilon_ci=_interval(bootstrap_epsilon, confidence),
            valid_bootstrap_replicates=len(bootstrap_delta),
        ),
        source_instance_count=len(source),
        target_instance_count=len(target),
        source_pair_count=len(source_differences),
        target_pair_count=len(target_differences),
        delta_source_percentile=delta_percentile,
        epsilon_source_percentile=epsilon_percentile,
    )


def calibrate_morphology_noise_axis(
    perturbation_magnitudes: Sequence[Real],
    *,
    bootstrap_replicates: int = 5000,
    seed: int = 20260823,
    confidence_level: float = 0.95,
) -> MorphologyNoiseAxis:
    """Estimate Q95/Q50 thresholds from per-image morphology perturbations."""

    values = _validated_values(
        perturbation_magnitudes, name="perturbation_magnitudes", minimum=2
    )
    replicate_count, resolved_seed, confidence = _validated_bootstrap(
        bootstrap_replicates, seed, confidence_level
    )
    point_delta = float(np.quantile(values, 0.95, method="higher"))
    point_epsilon = float(np.quantile(values, 0.50, method="higher"))
    rng = np.random.default_rng(resolved_seed)
    indices = rng.integers(0, len(values), size=(replicate_count, len(values)))
    resampled = values[indices]
    bootstrap_delta = np.quantile(resampled, 0.95, axis=1, method="higher")
    bootstrap_epsilon = np.quantile(resampled, 0.50, axis=1, method="higher")
    return MorphologyNoiseAxis(
        estimate=TransferEstimate(
            delta=point_delta,
            epsilon=point_epsilon,
            delta_ci=_interval(bootstrap_delta.tolist(), confidence),
            epsilon_ci=_interval(bootstrap_epsilon.tolist(), confidence),
            valid_bootstrap_replicates=replicate_count,
        ),
        instance_count=len(values),
    )


def calibrate_transfer_thresholds(
    source_burdens: Mapping[str, Sequence[Real]],
    target_burdens: Mapping[str, Sequence[Real]],
    morphology_magnitudes: Mapping[str, Sequence[Real]],
    primary_delta: Mapping[str, Real],
    primary_epsilon: Mapping[str, Real],
    *,
    bootstrap_replicates: int = 5000,
    seed: int = 20260823,
    confidence_level: float = 0.95,
) -> dict[str, dict[str, QuantileMatchedAxis | MorphologyNoiseAxis]]:
    """Calibrate both predeclared T_TRANSFER families for every configured axis."""

    axes = set(source_burdens)
    for name, mapping in (
        ("target_burdens", target_burdens),
        ("morphology_magnitudes", morphology_magnitudes),
        ("primary_delta", primary_delta),
        ("primary_epsilon", primary_epsilon),
    ):
        if set(mapping) != axes:
            raise ValueError(f"{name} axes must match source_burdens")
    if not axes:
        raise ValueError("at least one axis is required")
    return {
        axis: {
            "quantile_matched": calibrate_quantile_matched_axis(
                source_burdens[axis],
                target_burdens[axis],
                primary_delta=primary_delta[axis],
                primary_epsilon=primary_epsilon[axis],
                bootstrap_replicates=bootstrap_replicates,
                seed=seed,
                confidence_level=confidence_level,
            ),
            "morphology_noise_floor": calibrate_morphology_noise_axis(
                morphology_magnitudes[axis],
                bootstrap_replicates=bootstrap_replicates,
                seed=seed,
                confidence_level=confidence_level,
            ),
        }
        for axis in sorted(axes)
    }


__all__ = [
    "MorphologyNoiseAxis",
    "QuantileMatchedAxis",
    "TransferEstimate",
    "calibrate_morphology_noise_axis",
    "calibrate_quantile_matched_axis",
    "calibrate_transfer_thresholds",
]

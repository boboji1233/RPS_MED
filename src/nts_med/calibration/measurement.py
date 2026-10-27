"""Measurement-only threshold calibration from repeated reader annotations."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real

import numpy as np

from nts_med.relations.algebra import Interval


@dataclass(frozen=True, slots=True)
class AxisThresholdCalibration:
    """Frozen per-axis thresholds and their aggregate evidence counts."""

    delta: float
    epsilon: float
    max_width: float
    interval_half_width: float
    reader_deviation_q95: float
    instance_count: int
    reader_observation_count: int
    pairwise_difference_count: int


class OracleMode(StrEnum):
    """Predeclared ways to represent repeated-reader measurement evidence."""

    POINT_ORACLE_WITH_READER_TOLERANCE = "POINT_ORACLE_WITH_READER_TOLERANCE"
    READER_ENVELOPE_INTERVAL_ORACLE = "READER_ENVELOPE_INTERVAL_ORACLE"


@dataclass(frozen=True, slots=True)
class ThresholdInterval:
    """Percentile interval for one calibration threshold."""

    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class AxisThresholdUncertainty:
    """Case-resampling uncertainty and leave-one-case influence diagnostics."""

    delta_bootstrap_ci: ThresholdInterval
    epsilon_bootstrap_ci: ThresholdInterval
    delta_jackknife_range: ThresholdInterval
    epsilon_jackknife_range: ThresholdInterval
    maximum_delta_leave_one_case_change: float
    maximum_epsilon_leave_one_case_change: float
    most_influential_delta_case_index: int
    most_influential_epsilon_case_index: int
    case_count: int
    bootstrap_replicates: int
    seed: int


def _finite_values(values: Sequence[Real]) -> tuple[float, ...]:
    converted: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("reader burdens must be real numbers")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError("reader burdens must be finite and non-negative")
        converted.append(numeric)
    if len(converted) < 2:
        raise ValueError("each instance requires at least two reader burdens")
    return tuple(converted)


def _higher_quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("quantile evidence must not be empty")
    return float(np.quantile(np.asarray(values), quantile, method="higher"))


def calibrate_multireader_thresholds(
    burdens_by_axis: Mapping[str, Sequence[Sequence[Real]]],
) -> dict[str, AxisThresholdCalibration]:
    """Calibrate relation and interval tolerances without pair outcomes.

    ``delta`` is the higher-method 95th percentile of absolute pairwise reader
    differences and ``epsilon`` is their higher-method median. Expert-mask
    burdens remain point intervals, so both interval half-width and
    ``max_width`` are zero. This avoids counting reader variability once in an
    interval and again in ``delta``. The 95th percentile reader deviation is
    retained as a diagnostic for later measurement sensitivity work.
    """

    if not burdens_by_axis:
        raise ValueError("burdens_by_axis must not be empty")
    calibrated: dict[str, AxisThresholdCalibration] = {}
    for axis, instance_readings in burdens_by_axis.items():
        if not isinstance(axis, str) or not axis:
            raise ValueError("axis names must be non-empty strings")
        if not instance_readings:
            raise ValueError("each axis requires multireader instances")
        pairwise_differences: list[float] = []
        absolute_deviations: list[float] = []
        observation_count = 0
        for readings in instance_readings:
            values = _finite_values(readings)
            observation_count += len(values)
            pairwise_differences.extend(
                abs(left - right)
                for left, right in itertools.combinations(values, 2)
            )
            median = float(np.median(np.asarray(values)))
            absolute_deviations.extend(abs(value - median) for value in values)
        reader_deviation_q95 = _higher_quantile(absolute_deviations, 0.95)
        calibrated[axis] = AxisThresholdCalibration(
            delta=_higher_quantile(pairwise_differences, 0.95),
            epsilon=_higher_quantile(pairwise_differences, 0.50),
            max_width=0.0,
            interval_half_width=0.0,
            reader_deviation_q95=reader_deviation_q95,
            instance_count=len(instance_readings),
            reader_observation_count=observation_count,
            pairwise_difference_count=len(pairwise_differences),
        )
    return calibrated


def measurement_interval(value: Real, *, half_width: Real) -> Interval:
    """Create a non-negative burden interval using a frozen half-width."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("value must be a real number")
    if isinstance(half_width, bool) or not isinstance(half_width, Real):
        raise TypeError("half_width must be a real number")
    center = float(value)
    width = float(half_width)
    if not math.isfinite(center) or center < 0:
        raise ValueError("value must be finite and non-negative")
    if not math.isfinite(width) or width < 0:
        raise ValueError("half_width must be finite and non-negative")
    return Interval(max(0.0, center - width), center + width)


def point_oracle_interval(value: Real) -> Interval:
    """Represent an expert-mask burden as a point oracle."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("value must be a real number")
    center = float(value)
    if not math.isfinite(center) or center < 0:
        raise ValueError("value must be finite and non-negative")
    return Interval(center, center)


def reader_envelope_interval(values: Sequence[Real]) -> Interval:
    """Represent repeated-reader burdens by their observed closed envelope."""

    converted = _finite_values(values)
    return Interval(min(converted), max(converted))


def _percentile_interval(
    values: Sequence[float], confidence_level: float
) -> ThresholdInterval:
    alpha = (1.0 - confidence_level) / 2.0
    numeric = np.asarray(values, dtype=np.float64)
    return ThresholdInterval(
        lower=float(np.quantile(numeric, alpha, method="linear")),
        upper=float(np.quantile(numeric, 1.0 - alpha, method="linear")),
    )


def calibrate_threshold_uncertainty(
    burdens_by_axis: Mapping[str, Sequence[Sequence[Real]]],
    *,
    bootstrap_replicates: int = 2000,
    seed: int = 20260824,
    confidence_level: float = 0.95,
) -> dict[str, AxisThresholdUncertainty]:
    """Quantify calibration uncertainty by resampling whole reader cases.

    Bootstrap samples preserve all readings belonging to a selected case.
    Jackknife values are diagnostics only: no case is removed from the frozen
    calibration dataset or from any downstream analysis artifact.
    """

    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int):
        raise TypeError("bootstrap_replicates must be an integer")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(confidence_level, bool) or not isinstance(confidence_level, Real):
        raise TypeError("confidence_level must be a real number")
    confidence = float(confidence_level)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be finite and in (0, 1)")

    point = calibrate_multireader_thresholds(burdens_by_axis)
    rng = np.random.default_rng(seed)
    uncertainty: dict[str, AxisThresholdUncertainty] = {}
    for axis, raw_cases in burdens_by_axis.items():
        cases = tuple(_finite_values(case) for case in raw_cases)
        if len(cases) < 2:
            raise ValueError("threshold uncertainty requires at least two cases per axis")
        bootstrap_delta: list[float] = []
        bootstrap_epsilon: list[float] = []
        for _ in range(bootstrap_replicates):
            indices = rng.integers(0, len(cases), size=len(cases))
            resampled = tuple(cases[int(index)] for index in indices)
            estimate = calibrate_multireader_thresholds({axis: resampled})[axis]
            bootstrap_delta.append(estimate.delta)
            bootstrap_epsilon.append(estimate.epsilon)

        jackknife_delta: list[float] = []
        jackknife_epsilon: list[float] = []
        for omitted in range(len(cases)):
            retained = cases[:omitted] + cases[omitted + 1 :]
            estimate = calibrate_multireader_thresholds({axis: retained})[axis]
            jackknife_delta.append(estimate.delta)
            jackknife_epsilon.append(estimate.epsilon)
        delta_changes = [abs(value - point[axis].delta) for value in jackknife_delta]
        epsilon_changes = [
            abs(value - point[axis].epsilon) for value in jackknife_epsilon
        ]
        uncertainty[axis] = AxisThresholdUncertainty(
            delta_bootstrap_ci=_percentile_interval(bootstrap_delta, confidence),
            epsilon_bootstrap_ci=_percentile_interval(bootstrap_epsilon, confidence),
            delta_jackknife_range=ThresholdInterval(
                min(jackknife_delta), max(jackknife_delta)
            ),
            epsilon_jackknife_range=ThresholdInterval(
                min(jackknife_epsilon), max(jackknife_epsilon)
            ),
            maximum_delta_leave_one_case_change=max(delta_changes),
            maximum_epsilon_leave_one_case_change=max(epsilon_changes),
            most_influential_delta_case_index=delta_changes.index(max(delta_changes)),
            most_influential_epsilon_case_index=epsilon_changes.index(max(epsilon_changes)),
            case_count=len(cases),
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
    return uncertainty


__all__ = [
    "AxisThresholdCalibration",
    "AxisThresholdUncertainty",
    "OracleMode",
    "ThresholdInterval",
    "calibrate_multireader_thresholds",
    "calibrate_threshold_uncertainty",
    "measurement_interval",
    "point_oracle_interval",
    "reader_envelope_interval",
]

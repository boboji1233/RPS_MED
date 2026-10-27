"""Deterministic, model-free relation-persistence spectrum primitives.

This module deliberately accepts an explicit relation configuration at every
scale. It does not decide how a tolerance scale changes ``delta``, ``epsilon``
or interval widths. That scientific parameterisation belongs to the RPS-Med R1
preregistration and must be frozen before real-data analysis.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from types import MappingProxyType

from nts_med.relations import (
    Interval,
    Relation,
    RelationConfig,
    infer_relation_with_evidence,
)


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if converted < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return converted


@dataclass(frozen=True, slots=True)
class ScaleRelationConfig:
    """One explicitly supplied relation configuration at one tolerance scale."""

    scale: float
    config: RelationConfig

    def __post_init__(self) -> None:
        scale = _finite_nonnegative(self.scale, name="scale")
        if not isinstance(self.config, RelationConfig):
            raise TypeError("config must be a RelationConfig")
        object.__setattr__(self, "scale", scale)


class ToleranceScalePolicy(StrEnum):
    """Implemented scale families; callers must choose one explicitly."""

    MARGIN_ONLY = "MARGIN_ONLY"
    MARGIN_AND_EQUIVALENCE = "MARGIN_AND_EQUIVALENCE"


@dataclass(frozen=True, slots=True)
class SpectrumPoint:
    """Relation and privacy-safe axis evidence observed at one scale."""

    scale: float
    relation: Relation
    trigger_axes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CriticalToleranceBracket:
    """Sampled bracket containing a relation-state transition.

    This is not an analytical continuum root. The bracket only localises a
    transition between two adjacent, prespecified scales.
    """

    lower_scale: float
    upper_scale: float
    relation_before: Relation
    relation_after: Relation


@dataclass(frozen=True, slots=True)
class CriticalTolerancePoint:
    """An exact predicate boundary for the margin-only scale family."""

    scale: float
    relation_below: Relation | None
    relation_at: Relation
    relation_above: Relation | None


@dataclass(frozen=True, slots=True)
class SampledRelationPhase:
    """A maximal run of the same relation on the supplied scale grid."""

    start_scale: float
    end_scale: float
    relation: Relation
    sampled_point_count: int


@dataclass(frozen=True, slots=True)
class RelationDistributionPoint:
    """One source's aggregate relation distribution at one scale."""

    scale: float
    pair_count: int
    counts: Mapping[Relation, int]
    proportions: Mapping[Relation, float]


@dataclass(frozen=True, slots=True)
class CrossSourceStabilityPoint:
    """Descriptive agreement between two source distributions at one scale."""

    scale: float
    total_variation_distance: float
    agreement: float
    absolute_proportion_gaps: Mapping[Relation, float]


@dataclass(frozen=True, slots=True)
class IntegratedRelationStability:
    """Log-scale integrated cross-source distance and agreement."""

    log2_scale_span: float
    integrated_total_variation: float
    normalized_integrated_total_variation: float
    integrated_agreement: float


def build_scale_relation_configs(
    base: RelationConfig,
    scales: Sequence[float],
    policy: ToleranceScalePolicy,
) -> tuple[ScaleRelationConfig, ...]:
    """Build explicit configs for a declared scale policy without a default."""

    if not isinstance(base, RelationConfig):
        raise TypeError("base must be a RelationConfig")
    if not isinstance(policy, ToleranceScalePolicy):
        raise TypeError("policy must be a ToleranceScalePolicy")
    scale_values = tuple(
        _finite_nonnegative(value, name=f"scales[{index}]")
        for index, value in enumerate(scales)
    )
    if not scale_values:
        raise ValueError("scales must contain at least one value")
    if any(right <= left for left, right in pairwise(scale_values)):
        raise ValueError("scales must be strictly increasing")

    outputs: list[ScaleRelationConfig] = []
    for scale in scale_values:
        epsilon = (
            {axis: value * scale for axis, value in base.epsilon.items()}
            if policy is ToleranceScalePolicy.MARGIN_AND_EQUIVALENCE
            else dict(base.epsilon)
        )
        outputs.append(
            ScaleRelationConfig(
                scale=scale,
                config=RelationConfig(
                    delta={axis: value * scale for axis, value in base.delta.items()},
                    epsilon=epsilon,
                    max_width=dict(base.max_width),
                    numeric_tolerance=base.numeric_tolerance,
                ),
            )
        )
    return tuple(outputs)


def _validated_spectrum(points: Sequence[SpectrumPoint]) -> tuple[SpectrumPoint, ...]:
    values = tuple(points)
    if not values:
        raise ValueError("spectrum must contain at least one point")
    for index, point in enumerate(values):
        if not isinstance(point, SpectrumPoint):
            raise TypeError(f"spectrum[{index}] must be a SpectrumPoint")
        _finite_nonnegative(point.scale, name=f"spectrum[{index}].scale")
        if index and point.scale <= values[index - 1].scale:
            raise ValueError("spectrum scales must be strictly increasing")
    return values


def evaluate_relation_persistence(
    left: Mapping[str, Interval],
    right: Mapping[str, Interval],
    scale_configs: Sequence[ScaleRelationConfig],
) -> tuple[SpectrumPoint, ...]:
    """Evaluate one image-instance pair on an explicit ordered scale grid."""

    configs = tuple(scale_configs)
    if not configs:
        raise ValueError("scale_configs must contain at least one item")
    expected_axes: tuple[str, ...] | None = None
    points: list[SpectrumPoint] = []
    for index, item in enumerate(configs):
        if not isinstance(item, ScaleRelationConfig):
            raise TypeError(f"scale_configs[{index}] must be a ScaleRelationConfig")
        if index and item.scale <= configs[index - 1].scale:
            raise ValueError("scale values must be strictly increasing")
        if expected_axes is None:
            expected_axes = item.config.axes
        elif item.config.axes != expected_axes:
            raise ValueError("all scale configurations must use identical axes")
        evidence = infer_relation_with_evidence(left, right, item.config)
        points.append(
            SpectrumPoint(
                scale=item.scale,
                relation=evidence.relation,
                trigger_axes=evidence.trigger_axes,
            )
        )
    return tuple(points)


def critical_tolerance_brackets(
    spectrum: Sequence[SpectrumPoint],
) -> tuple[CriticalToleranceBracket, ...]:
    """Return adjacent scale brackets where the inferred relation changes."""

    points = _validated_spectrum(spectrum)
    return tuple(
        CriticalToleranceBracket(
            lower_scale=left.scale,
            upper_scale=right.scale,
            relation_before=left.relation,
            relation_after=right.relation,
        )
        for left, right in pairwise(points)
        if left.relation is not right.relation
    )


def analytic_margin_critical_tolerances(
    left: Mapping[str, Interval],
    right: Mapping[str, Interval],
    base: RelationConfig,
    *,
    minimum_scale: float,
    maximum_scale: float,
) -> tuple[CriticalTolerancePoint, ...]:
    """Locate exact relation predicate boundaries for ``MARGIN_ONLY`` scaling.

    Between the returned candidate boundaries every comparison used by the
    relation algebra is constant. Boundary states are evaluated separately
    because dominance is inclusive while crossing evidence is strict.
    """

    if not isinstance(base, RelationConfig):
        raise TypeError("base must be a RelationConfig")
    lower = _finite_nonnegative(minimum_scale, name="minimum_scale")
    upper = _finite_nonnegative(maximum_scale, name="maximum_scale")
    if lower >= upper:
        raise ValueError("minimum_scale must be less than maximum_scale")

    # Validate interval axes before deriving predicate boundaries.
    infer_relation_with_evidence(left, right, base)
    tolerance = base.numeric_tolerance
    candidates: set[float] = set()
    for axis in base.axes:
        delta = base.delta[axis]
        if delta == 0.0:
            continue
        directed_gaps = (
            right[axis].lower - left[axis].upper,
            left[axis].lower - right[axis].upper,
        )
        for gap in directed_gaps:
            for value in ((gap - tolerance) / delta, (gap + tolerance) / delta):
                if math.isfinite(value) and lower <= value <= upper:
                    candidates.add(value)

    ordered = tuple(sorted(candidates))
    partitions = (lower, *ordered, upper)

    def relation_at(scale: float) -> Relation:
        config = build_scale_relation_configs(
            base,
            (scale,),
            ToleranceScalePolicy.MARGIN_ONLY,
        )[0].config
        return infer_relation_with_evidence(left, right, config).relation

    outputs: list[CriticalTolerancePoint] = []
    for index, scale in enumerate(ordered, start=1):
        below_scale = None if scale == lower else (partitions[index - 1] + scale) / 2.0
        above_scale = None if scale == upper else (scale + partitions[index + 1]) / 2.0
        below = None if below_scale is None else relation_at(below_scale)
        at = relation_at(scale)
        above = None if above_scale is None else relation_at(above_scale)
        if (below is not None and below is not at) or (
            above is not None and above is not at
        ):
            outputs.append(
                CriticalTolerancePoint(
                    scale=scale,
                    relation_below=below,
                    relation_at=at,
                    relation_above=above,
                )
            )
    return tuple(outputs)


def sampled_relation_phases(
    spectrum: Sequence[SpectrumPoint],
) -> tuple[SampledRelationPhase, ...]:
    """Compress a sampled spectrum into maximal constant-relation phases."""

    points = _validated_spectrum(spectrum)
    phases: list[SampledRelationPhase] = []
    start = points[0]
    previous = points[0]
    point_count = 1
    for point in points[1:]:
        if point.relation is previous.relation:
            previous = point
            point_count += 1
            continue
        phases.append(
            SampledRelationPhase(
                start_scale=start.scale,
                end_scale=previous.scale,
                relation=start.relation,
                sampled_point_count=point_count,
            )
        )
        start = point
        previous = point
        point_count = 1
    phases.append(
        SampledRelationPhase(
            start_scale=start.scale,
            end_scale=previous.scale,
            relation=start.relation,
            sampled_point_count=point_count,
        )
    )
    return tuple(phases)


def aggregate_relation_phase_diagram(
    spectra: Sequence[Sequence[SpectrumPoint]],
) -> tuple[RelationDistributionPoint, ...]:
    """Aggregate pair spectra without persisting pair or instance identifiers."""

    validated = tuple(_validated_spectrum(spectrum) for spectrum in spectra)
    if not validated:
        raise ValueError("spectra must contain at least one pair spectrum")
    reference_scales = tuple(point.scale for point in validated[0])
    for index, spectrum in enumerate(validated[1:], start=1):
        if tuple(point.scale for point in spectrum) != reference_scales:
            raise ValueError(f"spectra[{index}] does not use the common scale grid")

    outputs: list[RelationDistributionPoint] = []
    pair_count = len(validated)
    for scale_index, scale in enumerate(reference_scales):
        observed = Counter(spectrum[scale_index].relation for spectrum in validated)
        counts = {relation: observed[relation] for relation in Relation}
        proportions = {
            relation: counts[relation] / pair_count for relation in Relation
        }
        outputs.append(
            RelationDistributionPoint(
                scale=scale,
                pair_count=pair_count,
                counts=MappingProxyType(counts),
                proportions=MappingProxyType(proportions),
            )
        )
    return tuple(outputs)


def cross_source_stability(
    left: Sequence[RelationDistributionPoint],
    right: Sequence[RelationDistributionPoint],
) -> tuple[CrossSourceStabilityPoint, ...]:
    """Compare source-level phase diagrams on an identical scale grid.

    ``agreement`` is ``1 - total_variation_distance``. It is descriptive only;
    whether it contributes to R1 is intentionally left for preregistration.
    """

    left_points = tuple(left)
    right_points = tuple(right)
    if not left_points or not right_points:
        raise ValueError("both source diagrams must contain at least one point")
    if len(left_points) != len(right_points):
        raise ValueError("source diagrams must have the same number of scales")

    outputs: list[CrossSourceStabilityPoint] = []
    for index, (left_point, right_point) in enumerate(
        zip(left_points, right_points, strict=True)
    ):
        if not isinstance(left_point, RelationDistributionPoint) or not isinstance(
            right_point, RelationDistributionPoint
        ):
            raise TypeError("source diagrams must contain RelationDistributionPoint values")
        if left_point.scale != right_point.scale:
            raise ValueError(f"source diagram scales differ at index {index}")
        gaps = {
            relation: abs(
                left_point.proportions[relation] - right_point.proportions[relation]
            )
            for relation in Relation
        }
        total_variation = 0.5 * sum(gaps.values())
        outputs.append(
            CrossSourceStabilityPoint(
                scale=left_point.scale,
                total_variation_distance=total_variation,
                agreement=1.0 - total_variation,
                absolute_proportion_gaps=MappingProxyType(gaps),
            )
        )
    return tuple(outputs)


def integrated_cross_source_stability(
    points: Sequence[CrossSourceStabilityPoint],
) -> IntegratedRelationStability:
    """Integrate pointwise total variation over ``log2(scale)``.

    The trapezoidal integral is normalized by the covered log-scale span, so
    the returned agreement remains in ``[0, 1]`` and grids with different
    resolutions are directly comparable over the same endpoints.
    """

    values = tuple(points)
    if len(values) < 2:
        raise ValueError("at least two stability points are required")
    log_scales: list[float] = []
    for index, point in enumerate(values):
        if not isinstance(point, CrossSourceStabilityPoint):
            raise TypeError(
                f"points[{index}] must be a CrossSourceStabilityPoint"
            )
        if point.scale <= 0.0:
            raise ValueError("stability scales must be positive")
        coordinate = math.log2(point.scale)
        if log_scales and coordinate <= log_scales[-1]:
            raise ValueError("stability scales must be strictly increasing")
        log_scales.append(coordinate)

    integral = sum(
        (right_u - left_u)
        * (left.total_variation_distance + right.total_variation_distance)
        / 2.0
        for (left_u, right_u), (left, right) in zip(
            pairwise(log_scales), pairwise(values), strict=True
        )
    )
    span = log_scales[-1] - log_scales[0]
    normalized = integral / span
    return IntegratedRelationStability(
        log2_scale_span=span,
        integrated_total_variation=integral,
        normalized_integrated_total_variation=normalized,
        integrated_agreement=1.0 - normalized,
    )


__all__ = [
    "CriticalToleranceBracket",
    "CriticalTolerancePoint",
    "CrossSourceStabilityPoint",
    "IntegratedRelationStability",
    "RelationDistributionPoint",
    "SampledRelationPhase",
    "ScaleRelationConfig",
    "SpectrumPoint",
    "ToleranceScalePolicy",
    "aggregate_relation_phase_diagram",
    "analytic_margin_critical_tolerances",
    "build_scale_relation_configs",
    "critical_tolerance_brackets",
    "cross_source_stability",
    "evaluate_relation_persistence",
    "integrated_cross_source_stability",
    "sampled_relation_phases",
]

"""Aggregate-only development analysis for the RPS-Med R1 programme.

This module accepts in-memory image-instance burdens and never persists pair
rows or instance identifiers. It is deliberately separate from formal R1
adjudication and model training.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations, pairwise
from typing import Any

import numpy as np

from nts_med.gate0.execution import SourceMeasurements
from nts_med.relations import Interval, Relation, RelationConfig, infer_relation
from rps_med.spectrum import (
    ToleranceScalePolicy,
    analytic_margin_critical_tolerances,
    build_scale_relation_configs,
)


def _point_intervals(values: Mapping[str, float]) -> dict[str, Interval]:
    return {axis: Interval(value, value) for axis, value in values.items()}


def _linear_quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("minimum", "q25", "median", "q75", "maximum")}
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(array, (0.0, 0.25, 0.5, 0.75, 1.0), method="linear")
    return {
        key: float(value)
        for key, value in zip(
            ("minimum", "q25", "median", "q75", "maximum"),
            quantiles,
            strict=True,
        )
    }


def _phase_payload(
    scales: Sequence[float], counts: Sequence[Counter[Relation]], pair_count: int
) -> list[dict[str, Any]]:
    return [
        {
            "scale": float(scale),
            "log2_scale": math.log2(scale),
            "pair_count": pair_count,
            "relation_counts": {relation.value: observed[relation] for relation in Relation},
            "relation_proportions": {
                relation.value: observed[relation] / pair_count for relation in Relation
            },
        }
        for scale, observed in zip(scales, counts, strict=True)
    ]


def _persistence_mass(points: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    coordinates = np.asarray([float(point["log2_scale"]) for point in points])
    span = float(coordinates[-1] - coordinates[0])
    return {
        relation.value: float(
            np.trapz(
                np.asarray(
                    [
                        float(
                            point["relation_proportions"][relation.value]
                        )
                        for point in points
                    ]
                ),
                coordinates,
            )
            / span
        )
        for relation in Relation
    }


def analyse_source(
    source: SourceMeasurements,
    *,
    base_config: RelationConfig,
    fine_scales: Sequence[float],
    coarse_stride: int,
    minimum_scale: float,
    maximum_scale: float,
) -> dict[str, Any]:
    """Compute one source's aggregate relation-persistence endpoints."""

    if source.reader is not None:
        raise ValueError("development source must not include reader-only measurements")
    if len(source.records) < 2:
        raise ValueError("development source requires at least two eligible records")
    if coarse_stride < 1 or (len(fine_scales) - 1) % coarse_stride:
        raise ValueError("coarse_stride must align exactly with the fine grid")
    fine_configs = build_scale_relation_configs(
        base_config,
        fine_scales,
        ToleranceScalePolicy.MARGIN_ONLY,
    )
    counts = [Counter[Relation]() for _ in fine_configs]
    transition_pair_count = 0
    sampled_transition_count = 0
    exact_transition_pair_count = 0
    exact_transition_count = 0
    exact_transition_scales: list[float] = []
    pair_count = 0

    for left_record, right_record in combinations(source.records, 2):
        pair_count += 1
        left = _point_intervals(left_record.primary)
        right = _point_intervals(right_record.primary)
        relations = tuple(
            infer_relation(left, right, item.config) for item in fine_configs
        )
        for index, relation in enumerate(relations):
            counts[index][relation] += 1
        changes = sum(
            left_relation is not right_relation
            for left_relation, right_relation in pairwise(relations)
        )
        sampled_transition_count += changes
        transition_pair_count += int(changes > 0)

        exact = analytic_margin_critical_tolerances(
            left,
            right,
            base_config,
            minimum_scale=minimum_scale,
            maximum_scale=maximum_scale,
        )
        exact_transition_count += len(exact)
        exact_transition_pair_count += int(bool(exact))
        exact_transition_scales.extend(point.scale for point in exact)

    fine_phase = _phase_payload(fine_scales, counts, pair_count)
    coarse_phase = fine_phase[::coarse_stride]
    fine_mass = _persistence_mass(fine_phase)
    coarse_mass = _persistence_mass(coarse_phase)
    mass_gaps = {
        relation.value: abs(fine_mass[relation.value] - coarse_mass[relation.value])
        for relation in Relation
    }
    return {
        "source": source.name,
        "role": source.role,
        "expected_input_count": source.expected_input_count,
        "observed_input_count": source.observed_input_count,
        "eligible_instance_count": len(source.records),
        "measurement_exclusion_count": source.measurement_exclusion_count,
        "pair_count": pair_count,
        "fine_phase_diagram": fine_phase,
        "coarse_phase_diagram": coarse_phase,
        "persistence_mass_fine": fine_mass,
        "persistence_mass_coarse": coarse_mass,
        "grid_stability": {
            "maximum_absolute_persistence_mass_gap": max(mass_gaps.values()),
            "absolute_persistence_mass_gaps": mass_gaps,
            "aligned_point_relation_count_mismatch": 0,
        },
        "sampled_transitions": {
            "pairs_with_transition": transition_pair_count,
            "pair_fraction_with_transition": transition_pair_count / pair_count,
            "transition_count": sampled_transition_count,
        },
        "analytic_critical_tolerance": {
            "pairs_with_transition": exact_transition_pair_count,
            "pair_fraction_with_transition": exact_transition_pair_count / pair_count,
            "transition_count": exact_transition_count,
            "scale_quantiles": _linear_quantiles(exact_transition_scales),
            "quantile_method": "LINEAR",
        },
    }


def cross_source_summary(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    """Compute normalized integrated TV from two aggregate phase diagrams."""

    left_points = left["fine_phase_diagram"]
    right_points = right["fine_phase_diagram"]
    if not isinstance(left_points, Sequence) or not isinstance(right_points, Sequence):
        raise TypeError("phase diagrams must be sequences")
    if len(left_points) != len(right_points) or len(left_points) < 2:
        raise ValueError("phase diagrams must share a non-trivial grid")
    coordinates: list[float] = []
    total_variations: list[float] = []
    pointwise: list[dict[str, float]] = []
    for left_point, right_point in zip(left_points, right_points, strict=True):
        if not isinstance(left_point, Mapping) or not isinstance(right_point, Mapping):
            raise TypeError("phase points must be mappings")
        if float(left_point["scale"]) != float(right_point["scale"]):
            raise ValueError("source scale grids differ")
        left_proportions = left_point["relation_proportions"]
        right_proportions = right_point["relation_proportions"]
        if not isinstance(left_proportions, Mapping) or not isinstance(
            right_proportions, Mapping
        ):
            raise TypeError("relation proportions must be mappings")
        gaps = [
            abs(
                float(left_proportions[relation.value])
                - float(right_proportions[relation.value])
            )
            for relation in Relation
        ]
        total_variation = 0.5 * sum(gaps)
        coordinate = float(left_point["log2_scale"])
        coordinates.append(coordinate)
        total_variations.append(total_variation)
        pointwise.append(
            {
                "scale": float(left_point["scale"]),
                "total_variation_distance": total_variation,
                "agreement": 1.0 - total_variation,
            }
        )
    span = coordinates[-1] - coordinates[0]
    integrated = float(
        np.trapz(np.asarray(total_variations), np.asarray(coordinates)) / span
    )
    return {
        "pointwise": pointwise,
        "log2_scale_span": span,
        "normalized_integrated_total_variation": integrated,
        "integrated_agreement": 1.0 - integrated,
    }


def base_relation_config(config: Mapping[str, Any]) -> RelationConfig:
    """Parse the frozen four-axis reference relation configuration."""

    thresholds = config.get("reference_thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("reference_thresholds must be a mapping")

    def values(name: str) -> dict[str, float]:
        raw = thresholds.get(name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"reference_thresholds.{name} must be a mapping")
        return {str(axis): float(value) for axis, value in raw.items()}

    return RelationConfig(
        delta=values("delta"),
        epsilon=values("epsilon"),
        max_width=values("max_width"),
        numeric_tolerance=float(thresholds.get("numeric_tolerance", 1.0e-12)),
    )


__all__ = ["analyse_source", "base_relation_config", "cross_source_summary"]

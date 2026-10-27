"""Aggregate-only formal R1 statistics for RPS-Med.

The executor precomputes source-local pair features in memory, then performs
image-instance bootstrap by rebuilding unordered pairs through multinomial
image counts. Pair rows, bootstrap rows and instance identifiers are never
returned or persisted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nts_med.gate0.execution import MeasurementRecord, SourceMeasurements
from nts_med.relations import Interval, Relation, RelationConfig, infer_relation
from rps_med.spectrum import (
    ToleranceScalePolicy,
    analytic_margin_critical_tolerances,
    build_scale_relation_configs,
)

_RELATIONS = tuple(Relation)
_RELATION_INDEX = {relation: index for index, relation in enumerate(_RELATIONS)}
_SI_INDEX = _RELATION_INDEX[Relation.STRUCTURAL_INCOMPARABLE]
_EQUIVALENT_INDEX = _RELATION_INDEX[Relation.EQUIVALENT]


@dataclass(frozen=True, slots=True)
class PairFeatureTable:
    """In-memory sufficient statistics for image-resampling bootstrap."""

    instance_count: int
    left_indices: NDArray[np.int32]
    right_indices: NDArray[np.int32]
    fine_relation_codes: NDArray[np.uint8]
    transition: NDArray[np.bool_]
    critical_scale: NDArray[np.float64]
    structural_incomparability_mass: NDArray[np.float64]
    primary_pattern_codes: NDArray[np.uint8]
    primary_pattern_inverse: NDArray[np.int64]
    critical_order: NDArray[np.int64]

    @property
    def pair_count(self) -> int:
        return int(self.left_indices.size)


def _point_intervals(values: Mapping[str, float]) -> dict[str, Interval]:
    return {axis: Interval(value, value) for axis, value in values.items()}


def _normalized_mass(indicator: NDArray[np.float64], coordinates: NDArray[np.float64]) -> float:
    return float(np.trapz(indicator, coordinates) / (coordinates[-1] - coordinates[0]))


def build_pair_feature_table(
    records: Sequence[MeasurementRecord],
    *,
    base_config: RelationConfig,
    fine_scales: Sequence[float],
) -> PairFeatureTable:
    """Precompute pair spectra and first analytic transition in memory."""

    values = tuple(records)
    if len(values) < 2:
        raise ValueError("formal R1 requires at least two source instances")
    scales = tuple(float(value) for value in fine_scales)
    if len(scales) < 3 or any(value <= 0.0 for value in scales):
        raise ValueError("fine scale grid must contain positive values")
    configs = build_scale_relation_configs(
        base_config, scales, ToleranceScalePolicy.MARGIN_ONLY
    )
    pair_indices = tuple(combinations(range(len(values)), 2))
    left_indices = np.asarray([left for left, _ in pair_indices], dtype=np.int32)
    right_indices = np.asarray([right for _, right in pair_indices], dtype=np.int32)
    relation_codes = np.empty((len(pair_indices), len(scales)), dtype=np.uint8)
    transitions = np.zeros(len(pair_indices), dtype=np.bool_)
    critical_scales = np.full(len(pair_indices), np.nan, dtype=np.float64)
    si_mass = np.zeros(len(pair_indices), dtype=np.float64)
    coordinates = np.log2(np.asarray(scales, dtype=np.float64))

    intervals = tuple(_point_intervals(record.primary) for record in values)
    for pair_index, (left_index, right_index) in enumerate(pair_indices):
        left = intervals[left_index]
        right = intervals[right_index]
        codes = np.asarray(
            [
                _RELATION_INDEX[infer_relation(left, right, item.config)]
                for item in configs
            ],
            dtype=np.uint8,
        )
        relation_codes[pair_index] = codes
        changes = np.flatnonzero(codes[1:] != codes[:-1])
        transitions[pair_index] = bool(changes.size)
        si_mass[pair_index] = _normalized_mass(
            np.asarray(codes == _SI_INDEX, dtype=np.float64), coordinates
        )
        critical = analytic_margin_critical_tolerances(
            left,
            right,
            base_config,
            minimum_scale=scales[0],
            maximum_scale=scales[-1],
        )
        if critical:
            critical_scales[pair_index] = critical[0].scale

    if not np.array_equal(transitions, np.isfinite(critical_scales)):
        raise ValueError("sampled and analytic transition states disagree")
    primary_patterns, primary_inverse = np.unique(
        relation_codes[:, ::2], axis=0, return_inverse=True
    )
    return PairFeatureTable(
        instance_count=len(values),
        left_indices=left_indices,
        right_indices=right_indices,
        fine_relation_codes=relation_codes,
        transition=transitions,
        critical_scale=critical_scales,
        structural_incomparability_mass=si_mass,
        primary_pattern_codes=np.asarray(primary_patterns, dtype=np.uint8),
        primary_pattern_inverse=np.asarray(primary_inverse, dtype=np.int64),
        critical_order=np.asarray(
            np.flatnonzero(np.isfinite(critical_scales))[
                np.argsort(
                    critical_scales[np.isfinite(critical_scales)], kind="stable"
                )
            ],
            dtype=np.int64,
        ),
    )


def _edge_weights(
    table: PairFeatureTable, instance_counts: NDArray[np.int64]
) -> tuple[NDArray[np.float64], int]:
    if instance_counts.shape != (table.instance_count,):
        raise ValueError("bootstrap instance counts have the wrong shape")
    if bool((instance_counts < 0).any()) or int(instance_counts.sum()) != table.instance_count:
        raise ValueError("bootstrap instance counts are invalid")
    weights = np.asarray(
        instance_counts[table.left_indices] * instance_counts[table.right_indices],
        dtype=np.float64,
    )
    self_pair_count = int(np.sum(instance_counts * (instance_counts - 1) // 2))
    return weights, self_pair_count


def _weighted_critical_median(
    table: PairFeatureTable, weights: NDArray[np.float64]
) -> float | None:
    if not table.critical_order.size:
        return None
    ordered_weights = weights[table.critical_order]
    if float(ordered_weights.sum()) <= 0.0:
        return None
    cumulative = np.cumsum(ordered_weights)
    target = float(cumulative[-1]) / 2.0
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(table.critical_scale[table.critical_order[index]])


def _weighted_phase_counts(
    table: PairFeatureTable,
    weights: NDArray[np.float64],
    self_pair_count: int,
    *,
    stride: int,
) -> NDArray[np.float64]:
    if stride == 2:
        patterns = table.primary_pattern_codes
        inverse = table.primary_pattern_inverse
    else:
        codes = table.fine_relation_codes[:, ::stride]
        patterns, inverse = np.unique(codes, axis=0, return_inverse=True)
    pattern_weights = np.bincount(inverse, weights=weights, minlength=len(patterns))
    outputs = np.zeros((patterns.shape[1], len(_RELATIONS)), dtype=np.float64)
    for scale_index in range(patterns.shape[1]):
        outputs[scale_index] = np.bincount(
            patterns[:, scale_index],
            weights=pattern_weights,
            minlength=len(_RELATIONS),
        )
        outputs[scale_index, _EQUIVALENT_INDEX] += self_pair_count
    return outputs


def _phase_proportions(
    table: PairFeatureTable,
    instance_counts: NDArray[np.int64],
    *,
    stride: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float | None]:
    weights, self_pair_count = _edge_weights(table, instance_counts)
    pair_count = table.instance_count * (table.instance_count - 1) // 2
    counts = _weighted_phase_counts(
        table, weights, self_pair_count, stride=stride
    )
    transition_fraction = float(np.dot(weights, table.transition) / pair_count)
    critical_median = _weighted_critical_median(table, weights)
    return counts / pair_count, counts, transition_fraction, critical_median


def _relation_masses(
    proportions: NDArray[np.float64], scales: NDArray[np.float64]
) -> dict[str, float]:
    coordinates = np.log2(scales)
    return {
        relation.value: _normalized_mass(proportions[:, index], coordinates)
        for index, relation in enumerate(_RELATIONS)
    }


def _integrated_agreement(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    scales: NDArray[np.float64],
) -> float:
    if left.shape != right.shape or left.shape[0] != scales.size:
        raise ValueError("cross-source phase diagrams are incompatible")
    total_variation = 0.5 * np.sum(np.abs(left - right), axis=1)
    return 1.0 - _normalized_mass(total_variation, np.log2(scales))


def _source_observed(
    source: SourceMeasurements,
    table: PairFeatureTable,
    fine_scales: NDArray[np.float64],
    *,
    coarse_stride: int,
) -> tuple[dict[str, Any], NDArray[np.float64], float | None]:
    counts = np.ones(table.instance_count, dtype=np.int64)
    fine_proportions, _, transition, critical = _phase_proportions(
        table, counts, stride=1
    )
    primary_proportions, primary_counts, _, _ = _phase_proportions(
        table, counts, stride=coarse_stride
    )
    fine_mass = _relation_masses(fine_proportions, fine_scales)
    primary_scales = fine_scales[::coarse_stride]
    primary_mass = _relation_masses(primary_proportions, primary_scales)
    gaps = {
        relation.value: abs(fine_mass[relation.value] - primary_mass[relation.value])
        for relation in _RELATIONS
    }
    phase = [
        {
            "scale": float(scale),
            "relation_counts": {
                relation.value: int(primary_counts[index, relation_index])
                for relation_index, relation in enumerate(_RELATIONS)
            },
            "relation_proportions": {
                relation.value: float(primary_proportions[index, relation_index])
                for relation_index, relation in enumerate(_RELATIONS)
            },
        }
        for index, scale in enumerate(primary_scales)
    ]
    return (
        {
            "source": source.name,
            "expected_input_count": source.expected_input_count,
            "observed_input_count": source.observed_input_count,
            "eligible_instance_count": len(source.records),
            "measurement_exclusion_count": source.measurement_exclusion_count,
            "pair_count": table.pair_count,
            "transition_fraction": transition,
            "critical_scale_median": critical,
            "persistence_mass_primary": primary_mass,
            "persistence_mass_sensitivity": fine_mass,
            "maximum_17_33_mass_gap": max(gaps.values()),
            "absolute_17_33_mass_gaps": gaps,
            "primary_phase_diagram": phase,
        },
        primary_proportions,
        critical,
    )


def _bootstrap_quantile(
    values: NDArray[np.float64], quantile: float
) -> float | None:
    if values.ndim != 1 or not bool(np.isfinite(values).all()):
        return None
    return float(np.quantile(values, quantile, method="linear"))


def execute_formal_r1(
    maples: SourceMeasurements,
    ddr: SourceMeasurements,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute frozen formal R1 statistics on already-authorized measurements."""

    if maples.measurement_exclusion_count or ddr.measurement_exclusion_count:
        raise ValueError("formal R1 requires zero measurement exclusions")
    if (
        len(maples.records) != maples.expected_input_count
        or len(ddr.records) != ddr.expected_input_count
    ):
        raise ValueError("formal R1 eligible counts must match frozen expectations")

    threshold_config = config.get("reference_thresholds")
    scale_config = config.get("scale")
    bootstrap_config = config.get("bootstrap")
    criteria_config = config.get("criteria")
    if not all(
        isinstance(value, Mapping)
        for value in (threshold_config, scale_config, bootstrap_config, criteria_config)
    ):
        raise ValueError("formal R1 configuration is incomplete")
    assert isinstance(threshold_config, Mapping)
    assert isinstance(scale_config, Mapping)
    assert isinstance(bootstrap_config, Mapping)
    assert isinstance(criteria_config, Mapping)

    def threshold_values(name: str) -> dict[str, float]:
        raw = threshold_config.get(name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"reference_thresholds.{name} is invalid")
        return {str(axis): float(value) for axis, value in raw.items()}

    relation_config = RelationConfig(
        delta=threshold_values("delta"),
        epsilon=threshold_values("epsilon"),
        max_width=threshold_values("max_width"),
        numeric_tolerance=float(threshold_config["numeric_tolerance"]),
    )
    minimum = float(scale_config["log2_minimum"])
    maximum = float(scale_config["log2_maximum"])
    fine_step = float(scale_config["sensitivity_step"])
    sensitivity_points = int(scale_config["sensitivity_points"])
    fine_scales = np.asarray(
        [2.0 ** (minimum + index * fine_step) for index in range(sensitivity_points)],
        dtype=np.float64,
    )
    coarse_stride = round(float(scale_config["primary_step"]) / fine_step)
    if (
        fine_scales.size != 33
        or fine_scales[::coarse_stride].size != 17
        or not math.isclose(math.log2(float(fine_scales[-1])), maximum)
    ):
        raise ValueError("formal R1 scale grids are invalid")
    scale_values = tuple(float(value) for value in fine_scales)

    maples_table = build_pair_feature_table(
        maples.records, base_config=relation_config, fine_scales=scale_values
    )
    ddr_table = build_pair_feature_table(
        ddr.records, base_config=relation_config, fine_scales=scale_values
    )
    maples_observed, maples_phase, maples_median = _source_observed(
        maples, maples_table, fine_scales, coarse_stride=coarse_stride
    )
    ddr_observed, ddr_phase, ddr_median = _source_observed(
        ddr, ddr_table, fine_scales, coarse_stride=coarse_stride
    )
    primary_scales = fine_scales[::coarse_stride]
    observed_agreement = _integrated_agreement(maples_phase, ddr_phase, primary_scales)
    observed_gap = (
        None
        if maples_median is None or ddr_median is None
        else abs(math.log2(maples_median) - math.log2(ddr_median))
    )

    replicates = int(bootstrap_config["replicates"])
    seed = int(bootstrap_config["seed"])
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, 6), dtype=np.float64)
    unestimable_gap_replicates = 0
    for replicate in range(replicates):
        maples_counts = rng.multinomial(
            maples_table.instance_count,
            np.full(maples_table.instance_count, 1.0 / maples_table.instance_count),
        ).astype(np.int64)
        ddr_counts = rng.multinomial(
            ddr_table.instance_count,
            np.full(ddr_table.instance_count, 1.0 / ddr_table.instance_count),
        ).astype(np.int64)
        maples_boot_phase, _, maples_transition, maples_boot_median = _phase_proportions(
            maples_table, maples_counts, stride=coarse_stride
        )
        ddr_boot_phase, _, ddr_transition, ddr_boot_median = _phase_proportions(
            ddr_table, ddr_counts, stride=coarse_stride
        )
        maples_mass = _relation_masses(maples_boot_phase, primary_scales)
        ddr_mass = _relation_masses(ddr_boot_phase, primary_scales)
        agreement = _integrated_agreement(
            maples_boot_phase, ddr_boot_phase, primary_scales
        )
        if maples_boot_median is None or ddr_boot_median is None:
            critical_gap = np.nan
            unestimable_gap_replicates += 1
        else:
            critical_gap = abs(
                math.log2(maples_boot_median) - math.log2(ddr_boot_median)
            )
        samples[replicate] = (
            maples_transition,
            ddr_transition,
            maples_mass[Relation.STRUCTURAL_INCOMPARABLE.value],
            ddr_mass[Relation.STRUCTURAL_INCOMPARABLE.value],
            agreement,
            critical_gap,
        )

    lower_q = float(bootstrap_config["lower_quantile"])
    upper_q = float(bootstrap_config["upper_quantile"])
    lower_names = (
        "maples_transition_fraction",
        "ddr_transition_fraction",
        "maples_structural_incomparability_mass",
        "ddr_structural_incomparability_mass",
        "cross_source_integrated_agreement",
    )
    lower_bounds = {
        name: _bootstrap_quantile(samples[:, index], lower_q)
        for index, name in enumerate(lower_names)
    }
    critical_gap_upper = _bootstrap_quantile(samples[:, 5], upper_q)

    checks = {
        "maples_transition_fraction": lower_bounds["maples_transition_fraction"] is not None
        and lower_bounds["maples_transition_fraction"]
        > float(criteria_config["transition_fraction_lower_strictly_above"]),
        "ddr_transition_fraction": lower_bounds["ddr_transition_fraction"] is not None
        and lower_bounds["ddr_transition_fraction"]
        > float(criteria_config["transition_fraction_lower_strictly_above"]),
        "maples_structural_incomparability_mass": lower_bounds[
            "maples_structural_incomparability_mass"
        ]
        is not None
        and lower_bounds["maples_structural_incomparability_mass"]
        > float(criteria_config["structural_incomparability_mass_lower_strictly_above"]),
        "ddr_structural_incomparability_mass": lower_bounds[
            "ddr_structural_incomparability_mass"
        ]
        is not None
        and lower_bounds["ddr_structural_incomparability_mass"]
        > float(criteria_config["structural_incomparability_mass_lower_strictly_above"]),
        "cross_source_integrated_agreement": lower_bounds[
            "cross_source_integrated_agreement"
        ]
        is not None
        and lower_bounds["cross_source_integrated_agreement"]
        > float(criteria_config["integrated_agreement_lower_strictly_above"]),
        "median_critical_scale_log2_gap": critical_gap_upper is not None
        and critical_gap_upper
        <= float(criteria_config["median_critical_scale_log2_gap_upper_at_most"]),
        "maples_grid_resolution": float(maples_observed["maximum_17_33_mass_gap"])
        <= float(criteria_config["maximum_17_33_mass_gap_at_most"]),
        "ddr_grid_resolution": float(ddr_observed["maximum_17_33_mass_gap"])
        <= float(criteria_config["maximum_17_33_mass_gap_at_most"]),
    }
    decision = "GO" if all(checks.values()) else "NO_GO"
    return {
        "schema_version": 1,
        "status": "FORMAL_RPS_MED_R1_COMPLETE",
        "decision": decision,
        "analysis_unit": "IMAGE_INSTANCE_PAIR",
        "sources": {
            "maples_dr_messidor1": maples_observed,
            "ddr": ddr_observed,
        },
        "cross_source": {
            "integrated_agreement": observed_agreement,
            "median_critical_scale_log2_gap": observed_gap,
        },
        "uncertainty": {
            "method": "IMAGE_INSTANCE_BOOTSTRAP_REBUILD_UNORDERED_PAIRS",
            "replicates": replicates,
            "seed": seed,
            "familywise_alpha": float(bootstrap_config["familywise_alpha"]),
            "stochastic_core_endpoint_count": int(
                bootstrap_config["stochastic_core_endpoint_count"]
            ),
            "lower_quantile": lower_q,
            "upper_quantile": upper_q,
            "lower_bounds": lower_bounds,
            "critical_gap_upper_bound": critical_gap_upper,
            "unestimable_critical_gap_replicates": unestimable_gap_replicates,
            "bootstrap_rows_persisted": False,
        },
        "criteria": {name: ("PASS" if passed else "FAIL") for name, passed in checks.items()},
        "source_pooling": False,
        "pair_rows_persisted": False,
        "formal_r1": "ADJUDICATED",
        "gate0": "NOT_RERUN",
        "training": "FORBIDDEN_NOT_RUN",
    }


__all__ = ["PairFeatureTable", "build_pair_feature_table", "execute_formal_r1"]

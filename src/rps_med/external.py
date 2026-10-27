"""Aggregate-only TJDR train/test external relation-persistence adjudication."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from nts_med.gate0.execution import SourceMeasurements
from nts_med.relations import Relation
from rps_med.development import base_relation_config
from rps_med.formal import (
    _bootstrap_quantile,
    _integrated_agreement,
    _phase_proportions,
    _relation_masses,
    _source_observed,
    build_pair_feature_table,
)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _scale_contract(config: Mapping[str, Any]) -> tuple[np.ndarray[Any, np.dtype[np.float64]], int]:
    scale = _mapping(config.get("scale"), name="scale")
    minimum = float(scale["log2_minimum"])
    maximum = float(scale["log2_maximum"])
    fine_step = float(scale["sensitivity_step"])
    points = int(scale["sensitivity_points"])
    values = np.asarray(
        [2.0 ** (minimum + index * fine_step) for index in range(points)],
        dtype=np.float64,
    )
    stride = round(float(scale["primary_step"]) / fine_step)
    if (
        values.size != 33
        or values[::stride].size != 17
        or not math.isclose(math.log2(float(values[-1])), maximum)
    ):
        raise ValueError("external gate scale contract is invalid")
    return values, stride


def execute_tjdr_external_gate(
    development: SourceMeasurements,
    test: SourceMeasurements,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Adjudicate one frozen TJDR official-test external replication gate."""

    for source in (development, test):
        if source.measurement_exclusion_count:
            raise ValueError("external gate requires zero measurement exclusions")
        if len(source.records) != source.expected_input_count:
            raise ValueError("external gate eligible count differs from expectation")
    bootstrap = _mapping(config.get("bootstrap"), name="bootstrap")
    criteria = _mapping(config.get("criteria"), name="criteria")
    fine_scales, stride = _scale_contract(config)
    base = base_relation_config(config)
    scale_values = tuple(float(value) for value in fine_scales)
    development_table = build_pair_feature_table(
        development.records, base_config=base, fine_scales=scale_values
    )
    test_table = build_pair_feature_table(
        test.records, base_config=base, fine_scales=scale_values
    )
    development_observed, development_phase, development_median = _source_observed(
        development, development_table, fine_scales, coarse_stride=stride
    )
    test_observed, test_phase, test_median = _source_observed(
        test, test_table, fine_scales, coarse_stride=stride
    )
    primary_scales = fine_scales[::stride]
    observed_agreement = _integrated_agreement(
        development_phase, test_phase, primary_scales
    )
    observed_gap = (
        None
        if development_median is None or test_median is None
        else abs(math.log2(development_median) - math.log2(test_median))
    )

    replicates = int(bootstrap["replicates"])
    seed = int(bootstrap["seed"])
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, 4), dtype=np.float64)
    unestimable_gap_replicates = 0
    for replicate in range(replicates):
        development_counts = rng.multinomial(
            development_table.instance_count,
            np.full(
                development_table.instance_count,
                1.0 / development_table.instance_count,
            ),
        ).astype(np.int64)
        test_counts = rng.multinomial(
            test_table.instance_count,
            np.full(test_table.instance_count, 1.0 / test_table.instance_count),
        ).astype(np.int64)
        development_boot_phase, _, _, development_boot_median = _phase_proportions(
            development_table, development_counts, stride=stride
        )
        test_boot_phase, _, test_transition, test_boot_median = _phase_proportions(
            test_table, test_counts, stride=stride
        )
        test_mass = _relation_masses(test_boot_phase, primary_scales)
        agreement = _integrated_agreement(
            development_boot_phase, test_boot_phase, primary_scales
        )
        if development_boot_median is None or test_boot_median is None:
            gap = np.nan
            unestimable_gap_replicates += 1
        else:
            gap = abs(math.log2(development_boot_median) - math.log2(test_boot_median))
        samples[replicate] = (
            test_transition,
            test_mass[Relation.STRUCTURAL_INCOMPARABLE.value],
            agreement,
            gap,
        )

    lower_quantile = float(bootstrap["lower_quantile"])
    upper_quantile = float(bootstrap["upper_quantile"])
    lower_bounds = {
        "test_transition_fraction": _bootstrap_quantile(samples[:, 0], lower_quantile),
        "test_structural_incomparability_mass": _bootstrap_quantile(
            samples[:, 1], lower_quantile
        ),
        "train_test_integrated_agreement": _bootstrap_quantile(
            samples[:, 2], lower_quantile
        ),
    }
    gap_upper = _bootstrap_quantile(samples[:, 3], upper_quantile)
    checks = {
        "test_transition_fraction": lower_bounds["test_transition_fraction"] is not None
        and lower_bounds["test_transition_fraction"]
        > float(criteria["transition_fraction_lower_strictly_above"]),
        "test_structural_incomparability_mass": lower_bounds[
            "test_structural_incomparability_mass"
        ]
        is not None
        and lower_bounds["test_structural_incomparability_mass"]
        > float(criteria["structural_incomparability_mass_lower_strictly_above"]),
        "train_test_integrated_agreement": lower_bounds[
            "train_test_integrated_agreement"
        ]
        is not None
        and lower_bounds["train_test_integrated_agreement"]
        > float(criteria["integrated_agreement_lower_strictly_above"]),
        "train_test_median_critical_scale_log2_gap": gap_upper is not None
        and gap_upper <= float(criteria["median_critical_scale_log2_gap_upper_at_most"]),
        "test_grid_resolution": float(test_observed["maximum_17_33_mass_gap"])
        <= float(criteria["maximum_17_33_mass_gap_at_most"]),
    }
    return {
        "schema_version": 1,
        "status": "FORMAL_RPS_MED_TJDR_EXTERNAL_GATE_COMPLETE",
        "decision": "GO" if all(checks.values()) else "NO_GO",
        "analysis_unit": "IMAGE_INSTANCE_PAIR",
        "sources": {
            "tjdr_official_train_frozen_reference": development_observed,
            "tjdr_official_test": test_observed,
        },
        "cross_split": {
            "integrated_agreement": observed_agreement,
            "median_critical_scale_log2_gap": observed_gap,
        },
        "uncertainty": {
            "method": "INDEPENDENT_IMAGE_INSTANCE_BOOTSTRAP_REBUILD_UNORDERED_PAIRS",
            "replicates": replicates,
            "seed": seed,
            "familywise_alpha": float(bootstrap["familywise_alpha"]),
            "stochastic_core_endpoint_count": int(
                bootstrap["stochastic_core_endpoint_count"]
            ),
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "lower_bounds": lower_bounds,
            "critical_gap_upper_bound": gap_upper,
            "unestimable_critical_gap_replicates": unestimable_gap_replicates,
            "bootstrap_rows_persisted": False,
        },
        "criteria": {name: ("PASS" if passed else "FAIL") for name, passed in checks.items()},
        "source_pooling": False,
        "pair_rows_persisted": False,
        "external_replication_gate": "ADJUDICATED",
        "gate0": "NOT_RERUN",
        "training": "FORBIDDEN_PENDING_RESULT_HANDOFF",
    }


__all__ = ["execute_tjdr_external_gate"]

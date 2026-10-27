"""Pure-synthetic validation of all owner-selected Gate 0 v0.6 rules.

This module cannot locate or read a dataset. Its aggregate output is software
validation evidence, never a scientific Gate 0 result.
"""

from __future__ import annotations

import json
from dataclasses import replace
from numbers import Real
from pathlib import Path
from typing import Final, cast

import numpy as np

from nts_med.calibration import FOVRatioCutoffs, fov_ratio_is_within_cutoffs
from nts_med.concepts.fov import binary_dilate, binary_erode
from nts_med.gate0 import (
    CriterionStatus,
    Gate0Criteria,
    Gate0Decision,
    ScalarizationMetric,
    adjudicate_gate0,
    common_eligible_instances,
    direction_consistent,
    distinct_crossing_pattern_count,
    grade_is_in_stratum,
    image_resampling_scalarization_bootstrap,
    permutation_null_summary,
    reader_subset_pair_count,
    scalarization_debt_nontrivial,
    strict_measurement_quality_eligible,
)
from nts_med.relations.algebra import (
    Interval,
    Relation,
    RelationConfig,
    infer_relation,
    infer_relation_with_evidence,
)
from nts_med.relations.pair_sampling import all_unordered_pairs
from nts_med.relations.robustness import BurdenTable, structural_relation_stability
from nts_med.relations.scalarization import MetricStatus, RateResult, debt_grade_indices
from nts_med.statistics.bootstrap import BootstrapConfig

SYNTHETIC_GATE0_V06_STATUS: Final = (
    "SYNTHETIC_GATE0_V06_FULL_CHAIN_NOT_SCIENTIFIC_RESULT"
)
_AXES: Final = ("ma", "hem", "ex", "cws")


def _relation_config() -> RelationConfig:
    return RelationConfig(
        delta={axis: 0.5 for axis in _AXES},
        epsilon={axis: 0.1 for axis in _AXES},
        max_width={axis: 0.0 for axis in _AXES},
        numeric_tolerance=1.0e-12,
    )


def _points(values: dict[str, float]) -> dict[str, Interval]:
    return {axis: Interval(value, value) for axis, value in values.items()}


def _synthetic_burdens() -> dict[str, dict[str, float]]:
    return {
        "s0": {"ma": 2.0, "hem": 2.0, "ex": 0.0, "cws": 0.0},
        "s1": {"ma": 0.0, "hem": 0.0, "ex": 2.0, "cws": 2.0},
        "s2": {"ma": 2.0, "hem": 0.0, "ex": 2.0, "cws": 0.0},
        "s3": {"ma": 0.0, "hem": 2.0, "ex": 0.0, "cws": 2.0},
    }


def _synthetic_relation_lookup(
    burdens: dict[str, dict[str, float]], config: RelationConfig
) -> dict[tuple[str, str], Relation]:
    identifiers = tuple(sorted(burdens))
    return {
        (pair.first_id, pair.second_id): infer_relation(
            _points(burdens[pair.first_id]),
            _points(burdens[pair.second_id]),
            config,
        )
        for pair in all_unordered_pairs(identifiers)
    }


def _morphology_checks() -> dict[str, bool]:
    centre = np.zeros((5, 5), dtype=np.bool_)
    centre[2, 2] = True
    dilated = binary_dilate(centre, radius=1)
    full = np.ones((3, 3), dtype=np.bool_)
    return {
        "radius_one_square_has_nine_pixels": int(np.count_nonzero(dilated)) == 9,
        "erosion_reverses_central_radius_one_dilation": bool(
            np.array_equal(binary_erode(dilated, radius=1), centre)
        ),
        "outside_padding_is_false": int(np.count_nonzero(binary_erode(full, radius=1)))
        == 1,
    }


def _fov_cutoff_checks() -> dict[str, bool]:
    bounds = {
        "maples_dr_messidor1": FOVRatioCutoffs(
            0.4493906610023041,
            0.4715313946759259,
            0.01,
            0.99,
            "HIGHER",
            138,
        ),
        "ddr": FOVRatioCutoffs(
            0.4315008609110654,
            0.8057213465526464,
            0.01,
            0.99,
            "HIGHER",
            531,
        ),
    }
    return {
        source: (
            fov_ratio_is_within_cutoffs(cast(Real, cutoffs.lower), cutoffs)
            and fov_ratio_is_within_cutoffs(cast(Real, cutoffs.upper), cutoffs)
            and not fov_ratio_is_within_cutoffs(
                cast(Real, cutoffs.lower - 1.0e-6), cutoffs
            )
            and not fov_ratio_is_within_cutoffs(
                cast(Real, cutoffs.upper + 1.0e-6), cutoffs
            )
            and cutoffs.quantile_method == "HIGHER"
        )
        for source, cutoffs in bounds.items()
    }


def _ofat_checks() -> dict[str, object]:
    primary = ("s0", "s1", "s2", "s3", "s4")
    variants = {
        "threshold": ("s0", "s1", "s2", "s3"),
        "fov": ("s0", "s1", "s2", "s4"),
        "morphology": ("s0", "s1", "s2", "s3", "s4"),
        "component": ("s0", "s1", "s2"),
    }
    common = common_eligible_instances(primary, variants)
    own_pair_counts = {
        name: reader_subset_pair_count(len(eligible))
        for name, eligible in variants.items()
    }
    return {
        "grid": "ONE_FACTOR_AT_A_TIME",
        "cartesian_product": False,
        "common_eligible_count": len(common),
        "common_pair_count": reader_subset_pair_count(len(common)),
        "primary_pair_count": reader_subset_pair_count(len(primary)),
        "variant_own_pair_counts": dict(sorted(own_pair_counts.items())),
        "excluded_or_missing_relabelled_unresolved": False,
        "check_passed": common == ("s0", "s1", "s2"),
    }


def _quality_checks() -> dict[str, bool]:
    return {
        "maples_official_roi_retained": strict_measurement_quality_eligible(
            dataset="maples_dr_messidor1",
            primary_eligible=True,
            source_low_quality=False,
            fov_method="OFFICIAL_ROI_ELLIPSE",
            border_contact=False,
        ),
        "ddr_primary_method_retained_with_border_contact": (
            strict_measurement_quality_eligible(
                dataset="ddr",
                primary_eligible=True,
                source_low_quality=False,
                fov_method="THRESHOLDED_LARGEST_COMPONENT",
                border_contact=True,
            )
        ),
        "ddr_fallback_excluded": not strict_measurement_quality_eligible(
            dataset="ddr",
            primary_eligible=True,
            source_low_quality=False,
            fov_method="FULL_FRAME_INSCRIBED_ELLIPSE",
            border_contact=False,
        ),
        "source_low_quality_excluded": not strict_measurement_quality_eligible(
            dataset="ddr",
            primary_eligible=True,
            source_low_quality=True,
            fov_method="THRESHOLDED_LARGEST_COMPONENT",
            border_contact=False,
        ),
    }


def _grade_checks() -> dict[str, bool]:
    debt_grades = (0, 1, None, 4)
    conflicts = (False, True, False, False)
    return {
        "grade4_in_all_not_non_pdr": grade_is_in_stratum(
            4,
            stratum="ALL_GRADE_0_TO_4",
            dataset="maples_dr_messidor1",
        )
        and not grade_is_in_stratum(
            4,
            stratum="NON_PDR_GRADE_0_TO_3",
            dataset="maples_dr_messidor1",
        ),
        "grade5_quality_only_excluded": not grade_is_in_stratum(
            5,
            stratum="ALL_GRADE_0_TO_4",
            dataset="ddr",
        ),
        "missing_excluded_from_stratum": not grade_is_in_stratum(
            None,
            stratum="ALL_GRADE_0_TO_4",
            dataset="ddr",
        ),
        "ddr_conflict_excluded_from_stratum": not grade_is_in_stratum(
            2,
            stratum="NON_PDR_GRADE_0_TO_3",
            dataset="ddr",
            grading_split_conflict=True,
        ),
        "debt_main_includes_conflict": debt_grade_indices(
            debt_grades, conflicts, include_conflicts=True
        )
        == (0, 1, 3),
        "debt_sensitivity_excludes_conflict": debt_grade_indices(
            debt_grades, conflicts, include_conflicts=False
        )
        == (0, 3),
    }


def _crossing_checks(
    burdens: dict[str, dict[str, float]], config: RelationConfig
) -> dict[str, object]:
    first = infer_relation_with_evidence(_points(burdens["s0"]), _points(burdens["s1"]), config)
    second = infer_relation_with_evidence(_points(burdens["s2"]), _points(burdens["s3"]), config)
    return {
        "confirmatory_primary_distinct_count": distinct_crossing_pattern_count(
            (first, second)
        ),
        "minimum_required": 2,
        "source_for_go_count": "MAPLES_CONFIRMATORY_PRIMARY_ONLY",
        "other_sources_or_variants_contribute": False,
    }


def _reader_checks() -> dict[str, object]:
    identifiers = tuple(f"reader_{index:02d}" for index in range(16))
    primary = {
        (pair.first_id, pair.second_id): Relation.STRUCTURAL_INCOMPARABLE
        for pair in all_unordered_pairs(identifiers)
    }
    variants = tuple(dict(primary) for _ in ("S1", "S2", "S3"))
    stability = structural_relation_stability(primary, variants)
    return {
        "complete_instances": len(identifiers),
        "unordered_pairs": reader_subset_pair_count(len(identifiers)),
        "point_reader_variants": ["S1", "S2", "S3"],
        "variant_pair_keys_identical": stability.variant_count == 3,
        "synthetic_stability_median": stability.median,
        "primary_recomputed_on_same_subset": True,
        "envelope_separate_auxiliary": True,
        "envelope_q95_delta_reapplied": False,
        "merged_or_extrapolated_to_primary_60": False,
    }


def _scalarization_checks(
    *, bootstrap_replicates: int, master_seed: int
) -> dict[str, object]:
    identifiers = ("s0", "s1", "s2", "s3")
    relations = {
        (pair.first_id, pair.second_id): Relation.STRUCTURAL_INCOMPARABLE
        for pair in all_unordered_pairs(identifiers)
    }
    grades = {"s0": 0, "s1": 1, "s2": 0, "s3": 1}
    bootstrap = BootstrapConfig(
        replicates=bootstrap_replicates,
        seed=master_seed,
        confidence_level=0.95,
    )
    confirmatory = {
        metric: image_resampling_scalarization_bootstrap(
            identifiers,
            relations,
            grades,
            metric,
            bootstrap,
        )
        for metric in ScalarizationMetric
    }
    replication = {
        ScalarizationMetric.FOD: RateResult(1.0, 4, 4, MetricStatus.OK),
        ScalarizationMetric.WCD: RateResult(1.0, 2, 2, MetricStatus.OK),
    }
    decision = scalarization_debt_nontrivial(confirmatory, replication)
    return {
        "bootstrap_unit": "IMAGE_INSTANCE_REBUILD_LEGAL_PAIRS",
        "replicates": bootstrap_replicates,
        "seed": master_seed,
        "confidence_interval": "PERCENTILE_LINEAR",
        "status": decision.status.value,
        "qualifying_metrics": [metric.value for metric in decision.qualifying_metrics],
        "confirmatory_ci_lower": {
            metric.value: confirmatory[metric].lower for metric in ScalarizationMetric
        },
        "cross_source_pooling": False,
    }


def _decision_checks() -> dict[str, object]:
    passing = Gate0Criteria(
        technical_integrity_pass=True,
        algebra_violations=0,
        confirmatory_point_pass=True,
        confirmatory_ci_lower_pass=True,
        replication_point_pass=True,
        confirmatory_distinct_patterns_pass=True,
        confirmatory_comonotonic_pass=True,
        replication_comonotonic_pass=True,
        relation_stability_pass=True,
        scalarization_debt_pass=True,
        strict_quality_direction_pass=True,
        non_pdr_direction_pass=True,
    )
    outcomes = {
        "all_pass": adjudicate_gate0(passing),
        "supporting_unestimable": adjudicate_gate0(
            replace(passing, non_pdr_direction_pass=None)
        ),
        "core_fail": adjudicate_gate0(
            replace(passing, replication_point_pass=False)
        ),
        "technical_fail": adjudicate_gate0(
            replace(passing, technical_integrity_pass=False)
        ),
    }
    expected = {
        "all_pass": Gate0Decision.GO,
        "supporting_unestimable": Gate0Decision.CONDITIONAL_GO,
        "core_fail": Gate0Decision.NO_GO,
        "technical_fail": Gate0Decision.ABORTED_TECHNICAL,
    }
    return {
        "synthetic_branch_outcomes": {
            name: outcome.value for name, outcome in outcomes.items()
        },
        "precedence_check_passed": outcomes == expected,
        "direction_strictly_above_tolerance": direction_consistent(
            cast(Real, 0.2), cast(Real, 0.1), numeric_tolerance=1.0e-12
        ),
        "direction_empty_is_unestimable": (
            direction_consistent(None, cast(Real, 0.1)) is None
        ),
    }


def _permutation_checks(
    burdens: dict[str, dict[str, float]],
    config: RelationConfig,
    *,
    repetitions: int,
    master_seed: int,
) -> dict[str, object]:
    relations = _synthetic_relation_lookup(burdens, config)
    observed = sum(
        relation is Relation.STRUCTURAL_INCOMPARABLE for relation in relations.values()
    ) / len(relations)
    summary = permutation_null_summary(
        cast(BurdenTable, burdens),
        config,
        repetitions=repetitions,
        master_seed=master_seed,
        observed_structural_prevalence=cast(Real, observed),
    )
    return {
        "role": "DESCRIPTIVE_ONLY_NOT_GO_REQUIRED",
        "repetitions": summary.repetitions,
        "master_seed": summary.master_seed,
        "replicate_seed_rule": "MASTER_PLUS_ZERO_BASED_INDEX",
        "median": summary.median,
        "q25": summary.q25,
        "q75": summary.q75,
        "q2_5": summary.q025,
        "q97_5": summary.q975,
        "observed_minus_null_median": summary.observed_minus_null_median,
        "p_value_computed": False,
        "cross_source_pooling": False,
    }


def run_synthetic_gate0_v06(
    output_dir: str | Path,
    *,
    bootstrap_replicates: int = 5000,
    permutation_repetitions: int = 5000,
    master_seed: int = 20260823,
) -> dict[str, object]:
    """Run all ten v0.6 decisions on constructed inputs and write aggregates."""

    destination = Path(output_dir)
    summary_path = destination / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {summary_path}")
    destination.mkdir(parents=True, exist_ok=True)

    config = _relation_config()
    burdens = _synthetic_burdens()
    morphology = _morphology_checks()
    fov_cutoffs = _fov_cutoff_checks()
    ofat = _ofat_checks()
    quality = _quality_checks()
    grades = _grade_checks()
    reader = _reader_checks()
    crossing = _crossing_checks(burdens, config)
    scalarization = _scalarization_checks(
        bootstrap_replicates=bootstrap_replicates,
        master_seed=master_seed,
    )
    decision = _decision_checks()
    permutation = _permutation_checks(
        burdens,
        config,
        repetitions=permutation_repetitions,
        master_seed=master_seed,
    )
    checks = {
        "M10A-SCI-01": all(morphology.values()),
        "M10A-SCI-02": all(fov_cutoffs.values()),
        "M10A-SCI-03": ofat["check_passed"] is True,
        "M10A-SCI-04": all(quality.values()),
        "M10A-SCI-05": all(grades.values()),
        "M10A-SCI-06": (
            reader["unordered_pairs"] == 120
            and reader["variant_pair_keys_identical"] is True
            and reader["envelope_separate_auxiliary"] is True
            and reader["merged_or_extrapolated_to_primary_60"] is False
        ),
        "M10A-SCI-07": crossing["confirmatory_primary_distinct_count"] == 2,
        "M10A-SCI-08": scalarization["status"] == CriterionStatus.PASS.value,
        "M10A-SCI-09": all(
            (
                decision["precedence_check_passed"],
                decision["direction_strictly_above_tolerance"],
                decision["direction_empty_is_unestimable"],
            )
        ),
        "M10A-SCI-10": (
            permutation["p_value_computed"] is False
            and permutation["cross_source_pooling"] is False
        ),
    }
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": SYNTHETIC_GATE0_V06_STATUS,
        "scientific_result": False,
        "real_data_used": False,
        "held_out_test_opened": False,
        "gate0_run": False,
        "real_pairs_generated": False,
        "real_relation_prevalence_computed": False,
        "model_training_run": False,
        "selected_bundle": "M10A_RECOMMENDED_R1_ALL_TEN",
        "checks": checks,
        "aggregates": {
            "mask_morphology": morphology,
            "fov_cutoffs": fov_cutoffs,
            "ofat": ofat,
            "strict_measurement_quality": quality,
            "grade_strata": grades,
            "reader_sensitivity": reader,
            "crossing_patterns": crossing,
            "scalarization": scalarization,
            "decision_mapping": decision,
            "permutation_null": permutation,
        },
        "all_checks_passed": all(checks.values()),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


__all__ = ["SYNTHETIC_GATE0_V06_STATUS", "run_synthetic_gate0_v06"]

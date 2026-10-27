"""Pure, fail-closed primitives for the owner-selected Gate 0 v0.6 rules.

The functions in this module do not locate datasets, read files, persist pair
rows, or authorize Gate 0. They are shared by synthetic validation and a later
separately authorized formal executor.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral, Real

import numpy as np

from nts_med.relations.algebra import (
    Interval,
    Relation,
    RelationConfig,
    RelationEvidence,
    infer_relation,
)
from nts_med.relations.pair_sampling import all_unordered_pairs
from nts_med.relations.robustness import BurdenTable, independently_permuted_burdens
from nts_med.relations.scalarization import MetricStatus, RateResult
from nts_med.statistics.bootstrap import BootstrapConfig, resampled_unordered_pair_weights


class CriterionStatus(StrEnum):
    """Three-valued status for a decision-supporting criterion."""

    PASS = "PASS"
    FAIL = "FAIL"
    UNESTIMABLE = "UNESTIMABLE"


class Gate0Decision(StrEnum):
    """Formal scientific decisions plus technical-abort precedence."""

    GO = "GO"
    CONDITIONAL_GO = "CONDITIONAL_GO"
    NO_GO = "NO_GO"
    ABORTED_TECHNICAL = "ABORTED_TECHNICAL"


class ScalarizationMetric(StrEnum):
    """Scalarization debt metrics admitted by preregistration v0.6."""

    FOD = "FOD"
    WCD = "WCD"


@dataclass(frozen=True, slots=True)
class ScalarizationBootstrapResult:
    """Image-resampling percentile interval for FOD or WCD."""

    metric: ScalarizationMetric
    point: RateResult
    lower: float | None
    upper: float | None
    valid_replicates: int
    degenerate_replicates: int
    requested_replicates: int
    seed: int
    confidence_level: float
    quantile_method: str


@dataclass(frozen=True, slots=True)
class ScalarizationDecision:
    """FOD-or-WCD nontriviality result across the two primary sources."""

    status: CriterionStatus
    qualifying_metrics: tuple[ScalarizationMetric, ...]
    unestimable_metrics: tuple[ScalarizationMetric, ...]


@dataclass(frozen=True, slots=True)
class Gate0Criteria:
    """Already-computed v0.6 criteria supplied to final adjudication."""

    technical_integrity_pass: bool
    algebra_violations: int
    confirmatory_point_pass: bool
    confirmatory_ci_lower_pass: bool
    replication_point_pass: bool
    confirmatory_distinct_patterns_pass: bool
    confirmatory_comonotonic_pass: bool
    replication_comonotonic_pass: bool
    relation_stability_pass: bool | None
    scalarization_debt_pass: bool | None
    strict_quality_direction_pass: bool | None
    non_pdr_direction_pass: bool | None


@dataclass(frozen=True, slots=True)
class PermutationNullSummary:
    """Aggregate-only independent-axis permutation-null summary."""

    repetitions: int
    master_seed: int
    median: float
    q25: float
    q75: float
    q025: float
    q975: float
    observed_minus_null_median: float | None


def _validated_identifiers(identifiers: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(identifiers, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of identifiers")
    values = tuple(identifiers)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} identifiers must be non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} identifiers must be unique")
    return tuple(sorted(values))


def common_eligible_instances(
    primary_eligible: Sequence[str],
    variant_eligible: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Return the common non-reader OFAT eligibility intersection."""

    primary = _validated_identifiers(primary_eligible, name="primary_eligible")
    if not variant_eligible:
        raise ValueError("at least one sensitivity variant is required")
    primary_set = set(primary)
    common = set(primary)
    for variant, identifiers in variant_eligible.items():
        if not isinstance(variant, str) or not variant:
            raise ValueError("variant names must be non-empty strings")
        selected = set(_validated_identifiers(identifiers, name=variant))
        if not selected <= primary_set:
            raise ValueError("variant eligibility must be a subset of primary eligibility")
        common &= selected
    return tuple(sorted(common))


def strict_measurement_quality_eligible(
    *,
    dataset: str,
    primary_eligible: bool,
    source_low_quality: bool,
    fov_method: str,
    border_contact: bool,
) -> bool:
    """Apply the selected source-specific strict measurement-quality rule."""

    for name, value in (
        ("primary_eligible", primary_eligible),
        ("source_low_quality", source_low_quality),
        ("border_contact", border_contact),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool")
    if not isinstance(fov_method, str) or not fov_method:
        raise ValueError("fov_method must be a non-empty string")
    if not primary_eligible or source_low_quality:
        return False
    if dataset == "maples_dr_messidor1":
        return fov_method == "OFFICIAL_ROI_ELLIPSE"
    if dataset == "ddr":
        return fov_method == "THRESHOLDED_LARGEST_COMPONENT"
    raise ValueError("strict quality is defined only for the two primary sources")


def grade_is_in_stratum(
    grade: int | None,
    *,
    stratum: str,
    dataset: str,
    grading_split_conflict: bool = False,
) -> bool:
    """Apply grade-only strata without changing primary lesion eligibility."""

    if not isinstance(grading_split_conflict, bool):
        raise TypeError("grading_split_conflict must be a bool")
    if dataset not in {"maples_dr_messidor1", "ddr"}:
        raise ValueError("grade strata are defined only for the two primary sources")
    if stratum not in {"ALL_GRADE_0_TO_4", "NON_PDR_GRADE_0_TO_3"}:
        raise ValueError("unknown grade stratum")
    if grade is None:
        return False
    if isinstance(grade, bool) or not isinstance(grade, Integral):
        raise TypeError("grade must be an integer or None")
    numeric = int(grade)
    if not 0 <= numeric <= 5:
        raise ValueError("grade must be in 0--5")
    if numeric == 5 or (dataset == "ddr" and grading_split_conflict):
        return False
    return numeric <= (4 if stratum == "ALL_GRADE_0_TO_4" else 3)


def reader_subset_pair_count(instance_count: int) -> int:
    """Return the complete unordered-pair denominator for a reader subset."""

    if isinstance(instance_count, bool) or not isinstance(instance_count, Integral):
        raise TypeError("instance_count must be an integer")
    count = int(instance_count)
    if count < 2:
        raise ValueError("reader sensitivity requires at least two instances")
    return count * (count - 1) // 2


CrossingPattern = tuple[tuple[str, ...], tuple[str, ...]]


def canonical_crossing_pattern(evidence: RelationEvidence) -> CrossingPattern:
    """Canonicalize worse-axis sets independently of pair orientation."""

    if not isinstance(evidence, RelationEvidence):
        raise TypeError("evidence must be RelationEvidence")
    if evidence.relation is not Relation.STRUCTURAL_INCOMPARABLE:
        raise ValueError("crossing patterns require structural incomparability")
    left = tuple(sorted(evidence.left_worse_axes))
    right = tuple(sorted(evidence.right_worse_axes))
    if not left or not right:
        raise ValueError("both worse-axis sets must be non-empty")
    if set(left) & set(right):
        raise ValueError("worse-axis sets must be disjoint")
    first, second = sorted((left, right))
    return first, second


def distinct_crossing_pattern_count(evidences: Sequence[RelationEvidence]) -> int:
    """Count unique canonical patterns in one confirmatory primary endpoint."""

    if isinstance(evidences, (str, bytes)):
        raise TypeError("evidences must be a sequence")
    return len({canonical_crossing_pattern(evidence) for evidence in evidences})


def _validated_probability(value: Real | None, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number or None")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return numeric


def direction_consistent(
    observed: Real | None,
    matched_comonotonic: Real | None,
    *,
    numeric_tolerance: float = 1.0e-12,
) -> bool | None:
    """Compare one observed stratum with its matched comonotonic null."""

    observed_value = _validated_probability(observed, name="observed")
    null_value = _validated_probability(matched_comonotonic, name="matched_comonotonic")
    if isinstance(numeric_tolerance, bool) or not isinstance(numeric_tolerance, Real):
        raise TypeError("numeric_tolerance must be a real number")
    tolerance = float(numeric_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("numeric_tolerance must be finite and non-negative")
    if observed_value is None or null_value is None:
        return None
    return observed_value - null_value > tolerance


def adjudicate_gate0(criteria: Gate0Criteria) -> Gate0Decision:
    """Apply technical-abort, core NO_GO, then supporting conditional precedence."""

    if not isinstance(criteria, Gate0Criteria):
        raise TypeError("criteria must be Gate0Criteria")
    if (
        isinstance(criteria.algebra_violations, bool)
        or not isinstance(criteria.algebra_violations, Integral)
        or criteria.algebra_violations < 0
    ):
        raise ValueError("algebra_violations must be a non-negative integer")
    if not criteria.technical_integrity_pass or criteria.algebra_violations != 0:
        return Gate0Decision.ABORTED_TECHNICAL
    core = (
        criteria.confirmatory_point_pass,
        criteria.confirmatory_ci_lower_pass,
        criteria.replication_point_pass,
        criteria.confirmatory_distinct_patterns_pass,
        criteria.confirmatory_comonotonic_pass,
        criteria.replication_comonotonic_pass,
    )
    if not all(core):
        return Gate0Decision.NO_GO
    supporting = (
        criteria.relation_stability_pass,
        criteria.scalarization_debt_pass,
        criteria.strict_quality_direction_pass,
        criteria.non_pdr_direction_pass,
    )
    if not all(value is True for value in supporting):
        return Gate0Decision.CONDITIONAL_GO
    return Gate0Decision.GO


def _validated_grade_lookup(
    instance_ids: tuple[str, ...], grades: Mapping[str, int]
) -> dict[str, int]:
    if set(grades) != set(instance_ids):
        raise ValueError("grades must cover every selected instance exactly")
    validated: dict[str, int] = {}
    for identifier, grade in grades.items():
        if (
            isinstance(grade, bool)
            or not isinstance(grade, Integral)
            or not 0 <= grade <= 4
        ):
            raise ValueError("scalarization grades must be integers in 0--4")
        validated[identifier] = int(grade)
    return validated


def _scalarization_counts(
    metric: ScalarizationMetric,
    relations: Mapping[tuple[str, str], Relation],
    grades: Mapping[str, int],
    weights: Mapping[tuple[str, str], int],
) -> tuple[int, int]:
    numerator = 0
    denominator = 0
    for pair, weight in weights.items():
        if weight < 0:
            raise ValueError("pair weights must be non-negative")
        left, right = pair
        eligible = (
            grades[left] != grades[right]
            if metric is ScalarizationMetric.FOD
            else grades[left] == grades[right]
        )
        if not eligible:
            continue
        denominator += weight
        if relations[pair] is Relation.STRUCTURAL_INCOMPARABLE:
            numerator += weight
    return numerator, denominator


def _rate_result(numerator: int, denominator: int) -> RateResult:
    if denominator == 0:
        return RateResult(None, 0, 0, MetricStatus.EMPTY_DENOMINATOR)
    return RateResult(numerator / denominator, numerator, denominator, MetricStatus.OK)


def image_resampling_scalarization_bootstrap(
    instance_ids: Sequence[str],
    relation_lookup: Mapping[tuple[str, str], Relation],
    grade_lookup: Mapping[str, int],
    metric: ScalarizationMetric,
    config: BootstrapConfig | None = None,
) -> ScalarizationBootstrapResult:
    """Recompute FOD or WCD after every legal image resample."""

    if not isinstance(metric, ScalarizationMetric):
        raise TypeError("metric must be ScalarizationMetric")
    resolved = config or BootstrapConfig()
    identifiers = _validated_identifiers(instance_ids, name="instance_ids")
    pairs = all_unordered_pairs(identifiers)
    if not pairs:
        raise ValueError("at least two instances are required")
    expected_keys = {(pair.first_id, pair.second_id) for pair in pairs}
    if set(relation_lookup) != expected_keys:
        raise ValueError("relation_lookup must cover every unordered pair exactly")
    if any(not isinstance(relation, Relation) for relation in relation_lookup.values()):
        raise TypeError("relation_lookup values must be Relation members")
    grades = _validated_grade_lookup(identifiers, grade_lookup)
    unit_weights = {pair: 1 for pair in expected_keys}
    point_numerator, point_denominator = _scalarization_counts(
        metric, relation_lookup, grades, unit_weights
    )
    point = _rate_result(point_numerator, point_denominator)
    if point.status is MetricStatus.EMPTY_DENOMINATOR:
        return ScalarizationBootstrapResult(
            metric=metric,
            point=point,
            lower=None,
            upper=None,
            valid_replicates=0,
            degenerate_replicates=int(resolved.replicates),
            requested_replicates=int(resolved.replicates),
            seed=int(resolved.seed),
            confidence_level=float(resolved.confidence_level),
            quantile_method="LINEAR",
        )

    estimates: list[float] = []
    degenerate = 0
    rng = random.Random(int(resolved.seed))
    for _ in range(int(resolved.replicates)):
        sampled = tuple(
            identifiers[rng.randrange(len(identifiers))] for _ in identifiers
        )
        weights = resampled_unordered_pair_weights(identifiers, sampled)
        numerator, denominator = _scalarization_counts(
            metric, relation_lookup, grades, weights
        )
        if denominator == 0:
            degenerate += 1
            continue
        estimates.append(numerator / denominator)
    if not estimates:
        return ScalarizationBootstrapResult(
            metric=metric,
            point=point,
            lower=None,
            upper=None,
            valid_replicates=0,
            degenerate_replicates=degenerate,
            requested_replicates=int(resolved.replicates),
            seed=int(resolved.seed),
            confidence_level=float(resolved.confidence_level),
            quantile_method="LINEAR",
        )
    alpha = (1.0 - float(resolved.confidence_level)) / 2.0
    return ScalarizationBootstrapResult(
        metric=metric,
        point=point,
        lower=float(np.quantile(estimates, alpha, method="linear")),
        upper=float(np.quantile(estimates, 1.0 - alpha, method="linear")),
        valid_replicates=len(estimates),
        degenerate_replicates=degenerate,
        requested_replicates=int(resolved.replicates),
        seed=int(resolved.seed),
        confidence_level=float(resolved.confidence_level),
        quantile_method="LINEAR",
    )


def scalarization_debt_nontrivial(
    confirmatory: Mapping[ScalarizationMetric, ScalarizationBootstrapResult],
    replication: Mapping[ScalarizationMetric, RateResult],
) -> ScalarizationDecision:
    """Apply the same-metric confirmatory-CI plus replication-event rule."""

    expected = set(ScalarizationMetric)
    if set(confirmatory) != expected or set(replication) != expected:
        raise ValueError("confirmatory and replication must contain FOD and WCD")
    qualifying: list[ScalarizationMetric] = []
    unestimable: list[ScalarizationMetric] = []
    for metric in ScalarizationMetric:
        primary = confirmatory[metric]
        external = replication[metric]
        if primary.metric is not metric:
            raise ValueError("confirmatory result metric key mismatch")
        if (
            primary.lower is None
            or primary.point.status is MetricStatus.EMPTY_DENOMINATOR
            or external.status is MetricStatus.EMPTY_DENOMINATOR
        ):
            unestimable.append(metric)
            continue
        if primary.lower > 0.0 and external.numerator > 0:
            qualifying.append(metric)
    if qualifying:
        status = CriterionStatus.PASS
    elif unestimable:
        status = CriterionStatus.UNESTIMABLE
    else:
        status = CriterionStatus.FAIL
    return ScalarizationDecision(status, tuple(qualifying), tuple(unestimable))


def _point_intervals(values: Mapping[str, Real]) -> dict[str, Interval]:
    return {axis: Interval(float(value), float(value)) for axis, value in values.items()}


def permutation_null_summary(
    burdens: BurdenTable,
    relation_config: RelationConfig,
    *,
    repetitions: int = 5000,
    master_seed: int = 20260823,
    observed_structural_prevalence: Real | None = None,
) -> PermutationNullSummary:
    """Summarize repeated independent-axis permutation prevalence, aggregate only."""

    if isinstance(repetitions, bool) or not isinstance(repetitions, Integral):
        raise TypeError("repetitions must be an integer")
    count = int(repetitions)
    if count < 1:
        raise ValueError("repetitions must be positive")
    if isinstance(master_seed, bool) or not isinstance(master_seed, Integral):
        raise TypeError("master_seed must be an integer")
    observed = _validated_probability(
        observed_structural_prevalence,
        name="observed_structural_prevalence",
    )
    identifiers = tuple(sorted(burdens))
    pairs = all_unordered_pairs(identifiers)
    if not pairs:
        raise ValueError("at least two burden rows are required")
    estimates: list[float] = []
    for index in range(count):
        permuted = independently_permuted_burdens(
            burdens,
            seed=int(master_seed) + index,
        )
        structural = sum(
            infer_relation(
                _point_intervals(permuted[pair.first_id]),
                _point_intervals(permuted[pair.second_id]),
                relation_config,
            )
            is Relation.STRUCTURAL_INCOMPARABLE
            for pair in pairs
        )
        estimates.append(structural / len(pairs))
    values = np.asarray(estimates, dtype=np.float64)
    median = float(np.quantile(values, 0.5, method="linear"))
    return PermutationNullSummary(
        repetitions=count,
        master_seed=int(master_seed),
        median=median,
        q25=float(np.quantile(values, 0.25, method="linear")),
        q75=float(np.quantile(values, 0.75, method="linear")),
        q025=float(np.quantile(values, 0.025, method="linear")),
        q975=float(np.quantile(values, 0.975, method="linear")),
        observed_minus_null_median=None if observed is None else observed - median,
    )


__all__ = [
    "CriterionStatus",
    "Gate0Criteria",
    "Gate0Decision",
    "PermutationNullSummary",
    "ScalarizationBootstrapResult",
    "ScalarizationDecision",
    "ScalarizationMetric",
    "adjudicate_gate0",
    "canonical_crossing_pattern",
    "common_eligible_instances",
    "direction_consistent",
    "distinct_crossing_pattern_count",
    "grade_is_in_stratum",
    "image_resampling_scalarization_bootstrap",
    "permutation_null_summary",
    "reader_subset_pair_count",
    "scalarization_debt_nontrivial",
    "strict_measurement_quality_eligible",
]

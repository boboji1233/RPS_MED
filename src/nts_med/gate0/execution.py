"""Aggregate-only formal Gate 0 v0.6 scientific adjudication.

This module consumes in-memory, opaque-identifier measurements. It never reads
datasets, writes pair rows, or authorizes execution.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, cast

from nts_med.gate0.methods import (
    CriterionStatus,
    Gate0Criteria,
    ScalarizationBootstrapResult,
    ScalarizationMetric,
    adjudicate_gate0,
    common_eligible_instances,
    direction_consistent,
    distinct_crossing_pattern_count,
    image_resampling_scalarization_bootstrap,
    permutation_null_summary,
    scalarization_debt_nontrivial,
)
from nts_med.relations.algebra import (
    Interval,
    Relation,
    RelationConfig,
    RelationEvidence,
    infer_relation_with_evidence,
)
from nts_med.relations.pair_sampling import all_unordered_pairs
from nts_med.relations.robustness import (
    BurdenTable,
    comonotonic_burdens,
    structural_relation_stability,
)
from nts_med.relations.scalarization import (
    MetricStatus,
    RateResult,
    forced_order_debt,
    within_grade_compression_debt,
)
from nts_med.statistics.bootstrap import (
    BootstrapConfig,
    image_resampling_relation_bootstrap,
)

BurdenVector = Mapping[str, float]
RelationTable = dict[tuple[str, str], Relation]


@dataclass(frozen=True, slots=True)
class MeasurementRecord:
    """One primary-eligible image measurement plus predeclared variants."""

    instance_id: str
    primary: BurdenVector
    variants: Mapping[str, BurdenVector | None]
    grade: int | None
    grading_split_conflict: bool
    strict_measurement_quality: bool


@dataclass(frozen=True, slots=True)
class ReaderMeasurements:
    """MAPLES reader-only point and envelope evidence on one fixed subset."""

    point_variants: Mapping[str, Mapping[str, BurdenVector]]
    envelopes: Mapping[str, Mapping[str, Interval]]


@dataclass(frozen=True, slots=True)
class SourceMeasurements:
    """One source's aggregate-safe formal input held only in memory."""

    name: str
    role: str
    expected_input_count: int
    observed_input_count: int
    records: tuple[MeasurementRecord, ...]
    measurement_exclusion_count: int
    reader: ReaderMeasurements | None = None


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return value


def _float_map(value: object, *, name: str) -> dict[str, float]:
    raw = _mapping(value, name=name)
    return {axis: float(raw[axis]) for axis in sorted(raw)}


def _relation_config(config: Mapping[str, Any]) -> RelationConfig:
    thresholds = _mapping(config.get("thresholds"), name="thresholds")
    return RelationConfig(
        delta=_float_map(thresholds.get("delta"), name="thresholds.delta"),
        epsilon=_float_map(thresholds.get("epsilon"), name="thresholds.epsilon"),
        max_width=_float_map(thresholds.get("max_width"), name="thresholds.max_width"),
        numeric_tolerance=float(
            _mapping(config.get("oracle"), name="oracle").get(
                "numeric_tolerance", 1.0e-12
            )
        ),
    )


def _point_intervals(values: BurdenVector) -> dict[str, Interval]:
    return {axis: Interval(float(value), float(value)) for axis, value in values.items()}


def _required_variant(record: MeasurementRecord, name: str) -> BurdenVector:
    value = record.variants.get(name)
    if value is None:
        raise ValueError(f"eligible variant {name} is missing")
    return value


def _required_grade(record: MeasurementRecord) -> int:
    if record.grade is None:
        raise ValueError("eligible formal severity grade is missing")
    return record.grade


def _as_real(value: float | None) -> Real | None:
    return None if value is None else cast(Real, value)


def _validated_records(
    source: SourceMeasurements,
    axes: tuple[str, ...],
) -> dict[str, MeasurementRecord]:
    if source.role not in {"CONFIRMATORY", "INDEPENDENT_REPLICATION"}:
        raise ValueError("source role is invalid")
    if source.expected_input_count < 1 or source.observed_input_count < 0:
        raise ValueError("source counts must be non-negative with positive expectation")
    if source.measurement_exclusion_count < 0:
        raise ValueError("measurement exclusion count must be non-negative")
    records = {record.instance_id: record for record in source.records}
    if len(records) != len(source.records):
        raise ValueError("source measurement identifiers must be unique")
    for record in source.records:
        if not record.instance_id or tuple(sorted(record.primary)) != axes:
            raise ValueError("primary burdens must use the frozen axes")
        for values in record.variants.values():
            if values is not None and tuple(sorted(values)) != axes:
                raise ValueError("variant burdens must use the frozen axes")
        if record.grade is not None and not 0 <= record.grade <= 4:
            raise ValueError("formal severity grades must be 0--4 or None")
    return records


def _relation_table(
    values: Mapping[str, BurdenVector],
    config: RelationConfig,
) -> tuple[RelationTable, tuple[RelationEvidence, ...], int]:
    identifiers = tuple(sorted(values))
    relations: RelationTable = {}
    evidences: list[RelationEvidence] = []
    violations = 0
    inverse = {
        Relation.LESS: Relation.GREATER,
        Relation.GREATER: Relation.LESS,
        Relation.EQUIVALENT: Relation.EQUIVALENT,
        Relation.STRUCTURAL_INCOMPARABLE: Relation.STRUCTURAL_INCOMPARABLE,
        Relation.UNRESOLVED: Relation.UNRESOLVED,
    }
    for pair in all_unordered_pairs(identifiers):
        left = _point_intervals(values[pair.first_id])
        right = _point_intervals(values[pair.second_id])
        evidence = infer_relation_with_evidence(left, right, config)
        reversed_evidence = infer_relation_with_evidence(right, left, config)
        violations += int(reversed_evidence.relation is not inverse[evidence.relation])
        relations[(pair.first_id, pair.second_id)] = evidence.relation
        evidences.append(evidence)
    return relations, tuple(evidences), violations


def _structural_rate(relations: Mapping[tuple[str, str], Relation]) -> RateResult:
    denominator = len(relations)
    numerator = sum(
        relation is Relation.STRUCTURAL_INCOMPARABLE
        for relation in relations.values()
    )
    if denominator == 0:
        return RateResult(None, 0, 0, MetricStatus.EMPTY_DENOMINATOR)
    return RateResult(numerator / denominator, numerator, denominator, MetricStatus.OK)


def _rate_payload(rate: RateResult) -> dict[str, object]:
    return {
        "value": rate.value,
        "numerator": rate.numerator,
        "denominator": rate.denominator,
        "status": rate.status.value,
    }


def _primary_endpoint(
    values: Mapping[str, BurdenVector],
    relations: RelationTable,
    bootstrap: BootstrapConfig,
) -> dict[str, object]:
    result = image_resampling_relation_bootstrap(tuple(values), relations, bootstrap)
    point = _structural_rate(relations)
    interval = result.confidence_intervals[Relation.STRUCTURAL_INCOMPARABLE]
    counts = Counter(relations.values())
    return {
        "eligible_instance_count": len(values),
        "unordered_pair_count": len(relations),
        "relation_counts": {
            relation.value: counts[relation] for relation in Relation
        },
        "structural_incomparability": _rate_payload(point),
        "structural_incomparability_ci_95": [interval.lower, interval.upper],
        "bootstrap_valid_replicates": result.valid_replicates,
        "bootstrap_degenerate_replicates": result.degenerate_replicates,
    }


def _comonotonic_rate(
    values: Mapping[str, BurdenVector],
    config: RelationConfig,
) -> RateResult:
    null_values = comonotonic_burdens(cast(BurdenTable, values))
    relations, _, _ = _relation_table(null_values, config)
    return _structural_rate(relations)


def _direction_for_subset(
    records: Mapping[str, MeasurementRecord],
    selected: Sequence[str],
    config: RelationConfig,
) -> tuple[bool | None, dict[str, object]]:
    identifiers = tuple(sorted(selected))
    if len(identifiers) < 2:
        return None, {
            "eligible_instance_count": len(identifiers),
            "unordered_pair_count": 0,
            "observed": None,
            "matched_comonotonic": None,
            "status": "UNESTIMABLE",
        }
    values = {identifier: records[identifier].primary for identifier in identifiers}
    relations, _, _ = _relation_table(values, config)
    observed = _structural_rate(relations)
    null = _comonotonic_rate(values, config)
    status = direction_consistent(_as_real(observed.value), _as_real(null.value))
    return status, {
        "eligible_instance_count": len(identifiers),
        "unordered_pair_count": len(relations),
        "observed": observed.value,
        "matched_comonotonic": null.value,
        "status": "PASS" if status is True else "FAIL" if status is False else "UNESTIMABLE",
    }


def _threshold_variant_configs(
    source_name: str,
    config: Mapping[str, Any],
    primary: RelationConfig,
) -> dict[str, RelationConfig]:
    thresholds = _mapping(config.get("thresholds"), name="thresholds")
    variants: dict[str, RelationConfig] = {}
    for multiplier in thresholds.get("scale_multipliers", []):
        numeric = float(multiplier)
        if numeric == 1.0:
            continue
        label = str(numeric).replace(".", "_")
        variants[f"THRESHOLD_SCALE_{label}"] = RelationConfig(
            delta={axis: value * numeric for axis, value in primary.delta.items()},
            epsilon=dict(primary.epsilon),
            max_width=dict(primary.max_width),
            numeric_tolerance=primary.numeric_tolerance,
        )
    if source_name == "ddr":
        for key in ("quantile_matched", "morphology_noise_floor"):
            transfer = _mapping(thresholds.get(key), name=f"thresholds.{key}")
            variants[f"THRESHOLD_TRANSFER_{key.upper()}"] = RelationConfig(
                delta=_float_map(transfer.get("delta"), name=f"{key}.delta"),
                epsilon=_float_map(transfer.get("epsilon"), name=f"{key}.epsilon"),
                max_width=dict(primary.max_width),
                numeric_tolerance=primary.numeric_tolerance,
            )
    return variants


def _variant_analysis(
    source: SourceMeasurements,
    records: Mapping[str, MeasurementRecord],
    config_payload: Mapping[str, Any],
    primary_config: RelationConfig,
) -> tuple[dict[str, object], float | None]:
    threshold_configs = _threshold_variant_configs(source.name, config_payload, primary_config)
    measurement_names = tuple(
        sorted({name for record in records.values() for name in record.variants})
    )
    decision_measurement_names = tuple(
        name for name in measurement_names if not name.startswith("AXIS_MEASURE_")
    )
    variant_eligible: dict[str, tuple[str, ...]] = {
        name: tuple(
            sorted(
                identifier
                for identifier, record in records.items()
                if record.variants.get(name) is not None
            )
        )
        for name in decision_measurement_names
    }
    for name in threshold_configs:
        variant_eligible[name] = tuple(sorted(records))
    common = common_eligible_instances(tuple(records), variant_eligible)
    common_primary_values = {identifier: records[identifier].primary for identifier in common}
    common_primary, _, _ = _relation_table(common_primary_values, primary_config)

    point_outputs: dict[str, object] = {}
    stability_tables: list[RelationTable] = []
    for name in sorted(variant_eligible):
        identifiers = variant_eligible[name]
        if name in threshold_configs:
            own_values = {identifier: records[identifier].primary for identifier in identifiers}
            own_config = threshold_configs[name]
            common_values = common_primary_values
        else:
            own_values = {
                identifier: _required_variant(records[identifier], name)
                for identifier in identifiers
            }
            own_config = primary_config
            common_values = {
                identifier: _required_variant(records[identifier], name)
                for identifier in common
            }
        own_relations, _, _ = _relation_table(own_values, own_config)
        common_relations, _, _ = _relation_table(common_values, own_config)
        stability_tables.append(common_relations)
        rate = _structural_rate(own_relations)
        point_outputs[name] = {
            "eligible_instance_count": len(identifiers),
            "attrition_from_primary": len(records) - len(identifiers),
            "unordered_pair_count": len(own_relations),
            "pair_attrition_from_primary": (
                len(records) * (len(records) - 1) // 2 - len(own_relations)
            ),
            "structural_incomparability": _rate_payload(rate),
        }

    descriptive: dict[str, object] = {}
    for name in measurement_names:
        if not name.startswith("AXIS_MEASURE_"):
            continue
        identifiers = tuple(
            sorted(
                identifier
                for identifier, record in records.items()
                if record.variants.get(name) is not None
            )
        )
        values = {
            identifier: _required_variant(records[identifier], name)
            for identifier in identifiers
        }
        relations, _, _ = _relation_table(values, primary_config)
        descriptive[name] = {
            "eligible_instance_count": len(identifiers),
            "unordered_pair_count": len(relations),
            "structural_incomparability": _rate_payload(_structural_rate(relations)),
            "decision_bearing": False,
        }

    stability_payload: dict[str, object]
    stability_median: float | None
    try:
        stability = structural_relation_stability(common_primary, stability_tables)
    except ValueError:
        stability_median = None
        stability_payload = {
            "status": "UNESTIMABLE",
            "common_eligible_instance_count": len(common),
            "common_unordered_pair_count": len(common_primary),
        }
    else:
        stability_median = stability.median
        stability_payload = {
            "status": "OK",
            "common_eligible_instance_count": len(common),
            "common_unordered_pair_count": len(common_primary),
            "primary_structural_pair_count": len(stability.pairs),
            "median": stability.median,
            "q25": stability.q25,
            "q75": stability.q75,
            "fraction_at_least_0_75": stability.fraction_at_least_0_75,
            "variant_count": stability.variant_count,
        }
    return {
        "grid": "ONE_FACTOR_AT_A_TIME",
        "own_denominator_variants": point_outputs,
        "axis_measure_descriptive": descriptive,
        "non_reader_stability": stability_payload,
        "excluded_or_missing_relabelled_unresolved": False,
        "primary_relation_table_persisted": False,
        "variant_relation_tables_persisted": False,
    }, stability_median


def _reader_analysis(
    source: SourceMeasurements,
    records: Mapping[str, MeasurementRecord],
    primary_config: RelationConfig,
) -> dict[str, object] | None:
    if source.reader is None:
        return None
    variant_ids = [set(values) for values in source.reader.point_variants.values()]
    envelope_ids = set(source.reader.envelopes)
    common = set(records) & envelope_ids
    for variant_set in variant_ids:
        common &= variant_set
    reader_identifiers = tuple(sorted(common))
    if len(reader_identifiers) != 16:
        raise ValueError("MAPLES reader scope must contain exactly 16 complete instances")
    primary_values = {
        identifier: records[identifier].primary for identifier in reader_identifiers
    }
    primary_relations, _, _ = _relation_table(primary_values, primary_config)
    reader_tables: list[RelationTable] = []
    for reader in ("S1", "S2", "S3"):
        values = source.reader.point_variants.get(reader)
        if values is None:
            raise ValueError("S1/S2/S3 reader variants are required")
        relations, _, _ = _relation_table(
            {identifier: values[identifier] for identifier in reader_identifiers},
            primary_config,
        )
        reader_tables.append(relations)
    try:
        stability = structural_relation_stability(primary_relations, reader_tables)
        stability_payload: dict[str, object] = {
            "status": "OK",
            "primary_structural_pair_count": len(stability.pairs),
            "median": stability.median,
            "q25": stability.q25,
            "q75": stability.q75,
            "fraction_at_least_0_75": stability.fraction_at_least_0_75,
        }
    except ValueError:
        stability_payload = {"status": "UNESTIMABLE"}

    envelope_config = RelationConfig(
        delta={axis: 0.0 for axis in primary_config.axes},
        epsilon=dict(primary_config.epsilon),
        max_width=dict(primary_config.max_width),
        numeric_tolerance=primary_config.numeric_tolerance,
    )
    envelope_relations: RelationTable = {}
    for pair in all_unordered_pairs(reader_identifiers):
        evidence = infer_relation_with_evidence(
            source.reader.envelopes[pair.first_id],
            source.reader.envelopes[pair.second_id],
            envelope_config,
        )
        envelope_relations[(pair.first_id, pair.second_id)] = evidence.relation
    return {
        "complete_instance_count": len(reader_identifiers),
        "unordered_pair_count": len(primary_relations),
        "point_variants": ["S1", "S2", "S3"],
        "stability": stability_payload,
        "envelope": {
            "role": "SEPARATE_AUXILIARY_INTERVAL_ORACLE",
            "reader_q95_delta_reapplied": False,
            "delta_used": 0.0,
            "relation_counts": {
                relation.value: sum(value is relation for value in envelope_relations.values())
                for relation in Relation
            },
        },
        "merged_or_extrapolated_to_primary_endpoint": False,
        "pair_rows_persisted": False,
    }


def _source_analysis(
    source: SourceMeasurements,
    config_payload: Mapping[str, Any],
    primary_config: RelationConfig,
    bootstrap: BootstrapConfig,
    *,
    permutation_repetitions: int,
    master_seed: int,
) -> tuple[dict[str, object], RelationTable, tuple[RelationEvidence, ...], int, float | None]:
    records = _validated_records(source, tuple(sorted(primary_config.axes)))
    values = {identifier: record.primary for identifier, record in records.items()}
    relations, evidences, violations = _relation_table(values, primary_config)
    endpoint = _primary_endpoint(values, relations, bootstrap)
    observed = _structural_rate(relations)
    null = _comonotonic_rate(values, primary_config)
    endpoint["matched_comonotonic"] = _rate_payload(null)
    endpoint["observed_exceeds_matched_comonotonic"] = direction_consistent(
        _as_real(observed.value), _as_real(null.value)
    )
    permutation = permutation_null_summary(
        cast(BurdenTable, values),
        primary_config,
        repetitions=permutation_repetitions,
        master_seed=master_seed,
        observed_structural_prevalence=_as_real(observed.value),
    )
    variants, stability_median = _variant_analysis(
        source, records, config_payload, primary_config
    )
    strict_direction, strict_payload = _direction_for_subset(
        records,
        tuple(
            identifier
            for identifier, record in records.items()
            if record.strict_measurement_quality
        ),
        primary_config,
    )
    non_pdr_direction, non_pdr_payload = _direction_for_subset(
        records,
        tuple(
            identifier
            for identifier, record in records.items()
            if record.grade is not None
            and record.grade <= 3
            and not record.grading_split_conflict
        ),
        primary_config,
    )
    all_grade_direction, all_grade_payload = _direction_for_subset(
        records,
        tuple(
            identifier
            for identifier, record in records.items()
            if record.grade is not None and not record.grading_split_conflict
        ),
        primary_config,
    )
    payload: dict[str, object] = {
        "role": source.role,
        "expected_input_count": source.expected_input_count,
        "observed_input_count": source.observed_input_count,
        "measurement_eligible_count": len(records),
        "measurement_exclusion_count": source.measurement_exclusion_count,
        "primary_endpoint": endpoint,
        "sensitivities": variants,
        "quality_strata": {
            "strict_measurement_quality": strict_payload,
            "strict_direction": strict_direction,
        },
        "grade_strata": {
            "all_grade_0_to_4": all_grade_payload,
            "all_grade_0_to_4_direction": all_grade_direction,
            "non_pdr_grade_0_to_3": non_pdr_payload,
            "non_pdr_direction": non_pdr_direction,
            "grade_used_for_primary_relation": False,
            "grading_split_conflicts_excluded": True,
        },
        "descriptive_permutation_null": {
            "repetitions": permutation.repetitions,
            "master_seed": permutation.master_seed,
            "median": permutation.median,
            "q25": permutation.q25,
            "q75": permutation.q75,
            "q2_5": permutation.q025,
            "q97_5": permutation.q975,
            "observed_minus_null_median": permutation.observed_minus_null_median,
            "p_value_computed": False,
            "decision_bearing": False,
        },
        "reader_sensitivity": _reader_analysis(source, records, primary_config),
        "algebra_violation_count": violations,
    }
    return payload, relations, evidences, violations, stability_median


def _scalar_rates(
    records: Mapping[str, MeasurementRecord],
    relations: RelationTable,
) -> dict[ScalarizationMetric, RateResult]:
    selected = tuple(
        sorted(identifier for identifier, record in records.items() if record.grade is not None)
    )
    pairs = all_unordered_pairs(selected)
    pair_relations = [relations[(pair.first_id, pair.second_id)] for pair in pairs]
    left = [_required_grade(records[pair.first_id]) for pair in pairs]
    right = [_required_grade(records[pair.second_id]) for pair in pairs]
    return {
        ScalarizationMetric.FOD: forced_order_debt(pair_relations, left, right),
        ScalarizationMetric.WCD: within_grade_compression_debt(
            pair_relations, left, right
        ),
    }


def _empty_scalar_bootstrap(
    metric: ScalarizationMetric,
    bootstrap: BootstrapConfig,
) -> ScalarizationBootstrapResult:
    return ScalarizationBootstrapResult(
        metric=metric,
        point=RateResult(None, 0, 0, MetricStatus.EMPTY_DENOMINATOR),
        lower=None,
        upper=None,
        valid_replicates=0,
        degenerate_replicates=int(bootstrap.replicates),
        requested_replicates=int(bootstrap.replicates),
        seed=int(bootstrap.seed),
        confidence_level=float(bootstrap.confidence_level),
        quantile_method="LINEAR",
    )


def _confirmatory_scalar_bootstrap(
    records: Mapping[str, MeasurementRecord],
    relations: RelationTable,
    bootstrap: BootstrapConfig,
) -> dict[ScalarizationMetric, ScalarizationBootstrapResult]:
    selected = tuple(
        sorted(identifier for identifier, record in records.items() if record.grade is not None)
    )
    if len(selected) < 2:
        return {
            metric: _empty_scalar_bootstrap(metric, bootstrap)
            for metric in ScalarizationMetric
        }
    grades = {identifier: _required_grade(records[identifier]) for identifier in selected}
    selected_relations = {
        (pair.first_id, pair.second_id): relations[(pair.first_id, pair.second_id)]
        for pair in all_unordered_pairs(selected)
    }
    return {
        metric: image_resampling_scalarization_bootstrap(
            selected, selected_relations, grades, metric, bootstrap
        )
        for metric in ScalarizationMetric
    }


def _combine_supporting(values: Sequence[bool | None]) -> bool | None:
    if any(value is False for value in values):
        return False
    if any(value is None for value in values):
        return None
    return True


def execute_gate0_v06(
    confirmatory: SourceMeasurements,
    replication: SourceMeasurements,
    config: Mapping[str, Any],
    *,
    bootstrap_replicates: int | None = None,
    permutation_repetitions: int | None = None,
) -> dict[str, object]:
    """Compute every v0.6 aggregate and return one formal adjudication."""

    if confirmatory.name != "maples_dr_messidor1" or confirmatory.role != "CONFIRMATORY":
        raise ValueError("the frozen confirmatory source is required")
    if replication.name != "ddr" or replication.role != "INDEPENDENT_REPLICATION":
        raise ValueError("the frozen independent replication source is required")

    primary_config = _relation_config(config)
    bootstrap_payload = _mapping(config.get("bootstrap"), name="bootstrap")
    bootstrap = BootstrapConfig(
        replicates=(
            int(bootstrap_payload.get("replicates", 5000))
            if bootstrap_replicates is None
            else bootstrap_replicates
        ),
        seed=int(bootstrap_payload.get("seed", 20260823)),
        confidence_level=float(bootstrap_payload.get("confidence_level", 0.95)),
    )
    nulls = _mapping(config.get("nulls"), name="nulls")
    descriptive = _mapping(nulls.get("descriptive"), name="nulls.descriptive")
    repetitions = (
        int(descriptive.get("repetitions_per_source", 5000))
        if permutation_repetitions is None
        else permutation_repetitions
    )
    master_seed = int(descriptive.get("master_seed", 20260823))

    confirm_payload, confirm_relations, confirm_evidence, confirm_violations, confirm_stability = (
        _source_analysis(
            confirmatory,
            config,
            primary_config,
            bootstrap,
            permutation_repetitions=repetitions,
            master_seed=master_seed,
        )
    )
    (
        replication_payload,
        replication_relations,
        replication_evidence,
        replication_violations,
        replication_stability,
    ) = _source_analysis(
        replication,
        config,
        primary_config,
        bootstrap,
        permutation_repetitions=repetitions,
        master_seed=master_seed,
    )
    confirm_records = _validated_records(confirmatory, tuple(sorted(primary_config.axes)))
    replication_records = _validated_records(
        replication, tuple(sorted(primary_config.axes))
    )
    confirm_scalar = _confirmatory_scalar_bootstrap(
        confirm_records, confirm_relations, bootstrap
    )
    replication_scalar = _scalar_rates(replication_records, replication_relations)
    replication_scalar_without_conflicts = _scalar_rates(
        {
            identifier: record
            for identifier, record in replication_records.items()
            if not record.grading_split_conflict
        },
        replication_relations,
    )
    scalar_decision = scalarization_debt_nontrivial(confirm_scalar, replication_scalar)
    scalar_pass = (
        True
        if scalar_decision.status is CriterionStatus.PASS
        else None
        if scalar_decision.status is CriterionStatus.UNESTIMABLE
        else False
    )

    confirm_endpoint = _mapping(
        confirm_payload["primary_endpoint"], name="confirmatory.primary_endpoint"
    )
    replication_endpoint = _mapping(
        replication_payload["primary_endpoint"], name="replication.primary_endpoint"
    )
    confirm_rate = _mapping(
        confirm_endpoint["structural_incomparability"], name="confirmatory.rate"
    )
    replication_rate = _mapping(
        replication_endpoint["structural_incomparability"], name="replication.rate"
    )
    confirm_ci = confirm_endpoint["structural_incomparability_ci_95"]
    if not isinstance(confirm_ci, list) or len(confirm_ci) != 2:
        raise ValueError("confirmatory confidence interval is invalid")
    pattern_count = distinct_crossing_pattern_count(
        tuple(
            evidence
            for evidence in confirm_evidence
            if evidence.relation is Relation.STRUCTURAL_INCOMPARABLE
        )
    )
    replication_pattern_count = distinct_crossing_pattern_count(
        tuple(
            evidence
            for evidence in replication_evidence
            if evidence.relation is Relation.STRUCTURAL_INCOMPARABLE
        )
    )
    decision_payload = _mapping(config.get("decision"), name="decision")
    core = _mapping(decision_payload.get("core_no_go"), name="decision.core_no_go")
    supporting = _mapping(
        decision_payload.get("supporting_conditional_if_fail_or_unestimable"),
        name="decision.supporting",
    )
    integrity = (
        confirmatory.observed_input_count == confirmatory.expected_input_count
        and replication.observed_input_count == replication.expected_input_count
        and len(confirm_records) >= 2
        and len(replication_records) >= 2
    )
    algebra_violations = confirm_violations + replication_violations
    relation_stability_pass = _combine_supporting(
        (
            None
            if confirm_stability is None
            else confirm_stability
            >= float(supporting["relation_stability_median_minimum"]),
            None
            if replication_stability is None
            else replication_stability
            >= float(supporting["relation_stability_median_minimum"]),
        )
    )
    strict_direction = _combine_supporting(
        (
            _mapping(confirm_payload["quality_strata"], name="confirm quality")[
                "strict_direction"
            ],
            _mapping(replication_payload["quality_strata"], name="rep quality")[
                "strict_direction"
            ],
        )
    )
    non_pdr_direction = _combine_supporting(
        (
            _mapping(confirm_payload["grade_strata"], name="confirm grades")[
                "non_pdr_direction"
            ],
            _mapping(replication_payload["grade_strata"], name="rep grades")[
                "non_pdr_direction"
            ],
        )
    )
    criteria = Gate0Criteria(
        technical_integrity_pass=integrity,
        algebra_violations=algebra_violations,
        confirmatory_point_pass=(
            confirm_rate.get("value") is not None
            and float(confirm_rate["value"])
            >= float(core["confirmatory_point_minimum"])
        ),
        confirmatory_ci_lower_pass=(
            float(confirm_ci[0]) > float(core["confirmatory_ci_lower_strictly_above"])
        ),
        replication_point_pass=(
            replication_rate.get("value") is not None
            and float(replication_rate["value"])
            >= float(core["replication_point_minimum"])
        ),
        confirmatory_distinct_patterns_pass=(
            pattern_count >= int(core["confirmatory_distinct_patterns_minimum"])
        ),
        confirmatory_comonotonic_pass=(
            confirm_endpoint["observed_exceeds_matched_comonotonic"] is True
        ),
        replication_comonotonic_pass=(
            replication_endpoint["observed_exceeds_matched_comonotonic"] is True
        ),
        relation_stability_pass=relation_stability_pass,
        scalarization_debt_pass=scalar_pass,
        strict_quality_direction_pass=strict_direction,
        non_pdr_direction_pass=non_pdr_direction,
    )
    decision = adjudicate_gate0(criteria)
    return {
        "schema_version": 1,
        "status": "FORMAL_GATE0_V06_COMPLETE",
        "scientific_result": True,
        "analysis_unit": "IMAGE_INSTANCE",
        "sources_pooled": False,
        "confirmatory": confirm_payload,
        "replication": replication_payload,
        "confirmatory_distinct_crossing_pattern_count": pattern_count,
        "replication_distinct_crossing_pattern_count": replication_pattern_count,
        "scalarization_debt": {
            "status": scalar_decision.status.value,
            "qualifying_metrics": [
                metric.value for metric in scalar_decision.qualifying_metrics
            ],
            "unestimable_metrics": [
                metric.value for metric in scalar_decision.unestimable_metrics
            ],
            "confirmatory": {
                metric.value: {
                    "point": _rate_payload(confirm_scalar[metric].point),
                    "ci_95": [confirm_scalar[metric].lower, confirm_scalar[metric].upper],
                    "valid_replicates": confirm_scalar[metric].valid_replicates,
                    "degenerate_replicates": confirm_scalar[metric].degenerate_replicates,
                }
                for metric in ScalarizationMetric
            },
            "replication": {
                metric.value: _rate_payload(replication_scalar[metric])
                for metric in ScalarizationMetric
            },
            "ddr_conflict_exclusion_sensitivity": {
                metric.value: _rate_payload(
                    replication_scalar_without_conflicts[metric]
                )
                for metric in ScalarizationMetric
            },
            "ddr_conflict_exclusion_decision_bearing": False,
            "cross_source_pooling": False,
        },
        "criteria": {
            field: getattr(criteria, field)
            for field in criteria.__dataclass_fields__
        },
        "decision": decision.value,
        "pair_rows_persisted": False,
        "source_identifiers_persisted": False,
        "absolute_paths_persisted": False,
        "training": "FORBIDDEN",
    }


__all__ = [
    "MeasurementRecord",
    "ReaderMeasurements",
    "SourceMeasurements",
    "execute_gate0_v06",
]

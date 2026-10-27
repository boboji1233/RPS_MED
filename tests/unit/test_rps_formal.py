from __future__ import annotations

from copy import deepcopy

from nts_med.gate0.execution import MeasurementRecord, SourceMeasurements
from rps_med.formal import execute_formal_r1


def _record(identifier: str, a: float, b: float) -> MeasurementRecord:
    return MeasurementRecord(
        instance_id=identifier,
        primary={"a": a, "b": b},
        variants={},
        grade=None,
        grading_split_conflict=False,
        strict_measurement_quality=True,
    )


def _source(name: str) -> SourceMeasurements:
    records = (
        _record(f"{name}_0", 0.0, 4.0),
        _record(f"{name}_1", 4.0, 0.0),
        _record(f"{name}_2", 0.0, 3.0),
        _record(f"{name}_3", 3.0, 0.0),
        _record(f"{name}_4", 1.0, 5.0),
        _record(f"{name}_5", 5.0, 1.0),
        _record(f"{name}_6", 0.5, 4.5),
        _record(f"{name}_7", 4.5, 0.5),
        _record(f"{name}_8", 1.5, 5.5),
        _record(f"{name}_9", 5.5, 1.5),
    )
    return SourceMeasurements(
        name=name,
        role="FORMAL_R1_CONFIRMATION",
        expected_input_count=len(records),
        observed_input_count=len(records),
        records=records,
        measurement_exclusion_count=0,
    )


def _config() -> dict[str, object]:
    return {
        "reference_thresholds": {
            "delta": {"a": 1.0, "b": 1.0},
            "epsilon": {"a": 0.1, "b": 0.1},
            "max_width": {"a": 0.0, "b": 0.0},
            "numeric_tolerance": 1.0e-12,
        },
        "scale": {
            "log2_minimum": -2.0,
            "log2_maximum": 2.0,
            "primary_step": 0.25,
            "sensitivity_step": 0.125,
            "primary_points": 17,
            "sensitivity_points": 33,
        },
        "bootstrap": {
            "replicates": 200,
            "seed": 20260825,
            "familywise_alpha": 0.05,
            "stochastic_core_endpoint_count": 6,
            "lower_quantile": 0.05 / 6.0,
            "upper_quantile": 1.0 - 0.05 / 6.0,
        },
        "criteria": {
            "transition_fraction_lower_strictly_above": -1.0,
            "structural_incomparability_mass_lower_strictly_above": -1.0,
            "integrated_agreement_lower_strictly_above": -1.0,
            "median_critical_scale_log2_gap_upper_at_most": 10.0,
            "maximum_17_33_mass_gap_at_most": 1.0,
        },
    }


def test_formal_r1_bootstrap_is_deterministic_aggregate_only_and_go() -> None:
    first = execute_formal_r1(_source("maples"), _source("ddr"), _config())
    second = execute_formal_r1(_source("maples"), _source("ddr"), _config())

    assert first == second
    assert first["decision"] == "GO"
    assert first["uncertainty"]["replicates"] == 200
    assert first["uncertainty"]["bootstrap_rows_persisted"] is False
    assert first["pair_rows_persisted"] is False
    serialized = str(first)
    assert "maples_0" not in serialized
    assert "ddr_0" not in serialized


def test_formal_r1_any_core_failure_is_no_go_without_training() -> None:
    config = deepcopy(_config())
    criteria = config["criteria"]
    assert isinstance(criteria, dict)
    criteria["transition_fraction_lower_strictly_above"] = 1.0

    result = execute_formal_r1(_source("maples"), _source("ddr"), config)

    assert result["decision"] == "NO_GO"
    assert result["criteria"]["maples_transition_fraction"] == "FAIL"
    assert result["training"] == "FORBIDDEN_NOT_RUN"


def test_formal_r1_measurement_exclusion_is_technical_error() -> None:
    source = _source("maples")
    excluded = SourceMeasurements(
        name=source.name,
        role=source.role,
        expected_input_count=source.expected_input_count,
        observed_input_count=source.observed_input_count,
        records=source.records,
        measurement_exclusion_count=1,
    )

    try:
        execute_formal_r1(excluded, _source("ddr"), _config())
    except ValueError as error:
        assert "zero measurement exclusions" in str(error)
    else:
        raise AssertionError("measurement exclusions must fail technically")

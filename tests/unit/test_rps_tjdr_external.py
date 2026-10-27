from __future__ import annotations

from nts_med.gate0.execution import MeasurementRecord, SourceMeasurements
from rps_med.external import execute_tjdr_external_gate


def _source(name: str, role: str) -> SourceMeasurements:
    values = tuple(
        {
            "ma": float(index),
            "hem": float(30 - index),
            "ex": float((index % 5) * 4),
            "cws": float((index % 3) * 6),
        }
        for index in range(20)
    )
    records = tuple(
        MeasurementRecord(
            instance_id=f"opaque-{name}-{index}",
            primary=value,
            variants={},
            grade=None,
            grading_split_conflict=False,
            strict_measurement_quality=True,
        )
        for index, value in enumerate(values)
    )
    return SourceMeasurements(name, role, len(records), len(records), records, 0)


def _config() -> dict[str, object]:
    return {
        "reference_thresholds": {
            "delta": {"ma": 2.0, "hem": 2.0, "ex": 2.0, "cws": 2.0},
            "epsilon": {"ma": 0.5, "hem": 0.5, "ex": 0.5, "cws": 0.5},
            "max_width": {"ma": 0.0, "hem": 0.0, "ex": 0.0, "cws": 0.0},
            "numeric_tolerance": 1e-12,
        },
        "scale": {
            "log2_minimum": -2.0,
            "log2_maximum": 2.0,
            "primary_step": 0.25,
            "sensitivity_step": 0.125,
            "sensitivity_points": 33,
        },
        "bootstrap": {
            "replicates": 40,
            "seed": 7,
            "familywise_alpha": 0.05,
            "stochastic_core_endpoint_count": 4,
            "lower_quantile": 0.0125,
            "upper_quantile": 0.9875,
        },
        "criteria": {
            "transition_fraction_lower_strictly_above": -1.0,
            "structural_incomparability_mass_lower_strictly_above": -1.0,
            "integrated_agreement_lower_strictly_above": -1.0,
            "median_critical_scale_log2_gap_upper_at_most": 10.0,
            "maximum_17_33_mass_gap_at_most": 1.0,
        },
    }


def test_external_gate_returns_aggregate_simultaneous_endpoints() -> None:
    result = execute_tjdr_external_gate(
        _source("train", "DEVELOPMENT_CALIBRATION"),
        _source("test", "INDEPENDENT_EXTERNAL_REPLICATION"),
        _config(),
    )

    assert result["decision"] == "GO"
    assert set(result["criteria"]) == {
        "test_transition_fraction",
        "test_structural_incomparability_mass",
        "train_test_integrated_agreement",
        "train_test_median_critical_scale_log2_gap",
        "test_grid_resolution",
    }
    assert result["uncertainty"]["stochastic_core_endpoint_count"] == 4
    assert result["uncertainty"]["bootstrap_rows_persisted"] is False
    assert result["pair_rows_persisted"] is False


def test_external_gate_fails_closed_on_measurement_exclusion() -> None:
    train = _source("train", "DEVELOPMENT_CALIBRATION")
    test = _source("test", "INDEPENDENT_EXTERNAL_REPLICATION")
    invalid = SourceMeasurements(
        test.name,
        test.role,
        test.expected_input_count,
        test.observed_input_count,
        test.records,
        1,
    )

    import pytest

    with pytest.raises(ValueError, match="zero measurement exclusions"):
        execute_tjdr_external_gate(train, invalid, _config())

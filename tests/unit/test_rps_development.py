from __future__ import annotations

import pytest

from nts_med.gate0.execution import MeasurementRecord, SourceMeasurements
from nts_med.relations import RelationConfig
from rps_med.development import analyse_source, cross_source_summary


def _record(identifier: str, a: float, b: float) -> MeasurementRecord:
    return MeasurementRecord(
        instance_id=identifier,
        primary={"a": a, "b": b},
        variants={},
        grade=None,
        grading_split_conflict=False,
        strict_measurement_quality=True,
    )


def _source(name: str, records: tuple[MeasurementRecord, ...]) -> SourceMeasurements:
    return SourceMeasurements(
        name=name,
        role="DEVELOPMENT_CALIBRATION",
        expected_input_count=len(records),
        observed_input_count=len(records),
        records=records,
        measurement_exclusion_count=0,
    )


def test_development_analysis_is_aggregate_only_and_grid_stable() -> None:
    scales = tuple(2.0 ** (-2.0 + index * 0.125) for index in range(33))
    config = RelationConfig(
        delta={"a": 1.0, "b": 1.0},
        epsilon={"a": 0.1, "b": 0.1},
        max_width={"a": 0.0, "b": 0.0},
    )
    source = _source(
        "synthetic_a",
        (
            _record("private_a", 3.0, 0.0),
            _record("private_b", 0.0, 3.0),
            _record("private_c", 1.0, 1.0),
        ),
    )

    result = analyse_source(
        source,
        base_config=config,
        fine_scales=scales,
        coarse_stride=2,
        minimum_scale=0.25,
        maximum_scale=4.0,
    )

    assert result["pair_count"] == 3
    assert len(result["fine_phase_diagram"]) == 33
    assert len(result["coarse_phase_diagram"]) == 17
    assert result["analytic_critical_tolerance"]["pairs_with_transition"] > 0
    assert "private_a" not in str(result)
    assert sum(result["persistence_mass_fine"].values()) == pytest.approx(1.0)

    stability = cross_source_summary(result, result)
    assert stability["normalized_integrated_total_variation"] == 0.0
    assert stability["integrated_agreement"] == 1.0


def test_development_analysis_requires_nested_grid_and_two_records() -> None:
    config = RelationConfig(
        delta={"a": 1.0}, epsilon={"a": 0.1}, max_width={"a": 0.0}
    )
    with pytest.raises(ValueError, match="two eligible"):
        analyse_source(
            _source("small", (_record("one", 0.0, 0.0),)),
            base_config=config,
            fine_scales=(0.25, 0.5, 1.0),
            coarse_stride=2,
            minimum_scale=0.25,
            maximum_scale=1.0,
        )

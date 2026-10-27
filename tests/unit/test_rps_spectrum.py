from __future__ import annotations

import pytest

from nts_med.relations import Interval, Relation, RelationConfig
from rps_med import (
    ScaleRelationConfig,
    SpectrumPoint,
    ToleranceScalePolicy,
    aggregate_relation_phase_diagram,
    analytic_margin_critical_tolerances,
    build_scale_relation_configs,
    critical_tolerance_brackets,
    cross_source_stability,
    evaluate_relation_persistence,
    integrated_cross_source_stability,
    sampled_relation_phases,
)


def _config(delta: float) -> RelationConfig:
    return RelationConfig(
        delta={"a": delta, "b": delta},
        epsilon={"a": 0.1, "b": 0.1},
        max_width={"a": 0.1, "b": 0.1},
    )


def _point(value: float) -> Interval:
    return Interval(value, value)


def test_relation_persistence_records_a_sampled_transition() -> None:
    spectrum = evaluate_relation_persistence(
        {"a": _point(3.0), "b": _point(0.0)},
        {"a": _point(0.0), "b": _point(3.0)},
        (
            ScaleRelationConfig(0.0, _config(0.0)),
            ScaleRelationConfig(1.0, _config(1.0)),
            ScaleRelationConfig(4.0, _config(4.0)),
        ),
    )

    assert [point.relation for point in spectrum] == [
        Relation.STRUCTURAL_INCOMPARABLE,
        Relation.STRUCTURAL_INCOMPARABLE,
        Relation.UNRESOLVED,
    ]
    assert spectrum[0].trigger_axes == ("a", "b")


def test_scale_policy_must_be_explicit_and_preserves_width_contract() -> None:
    base = _config(2.0)
    margin_only = build_scale_relation_configs(
        base,
        (0.5, 1.0, 2.0),
        ToleranceScalePolicy.MARGIN_ONLY,
    )
    coupled = build_scale_relation_configs(
        base,
        (0.5, 1.0, 2.0),
        ToleranceScalePolicy.MARGIN_AND_EQUIVALENCE,
    )

    assert margin_only[0].config.delta["a"] == 1.0
    assert margin_only[0].config.epsilon["a"] == base.epsilon["a"]
    assert coupled[0].config.epsilon["a"] == base.epsilon["a"] * 0.5
    assert coupled[-1].config.max_width == base.max_width

    with pytest.raises(TypeError, match="policy"):
        build_scale_relation_configs(base, (1.0,), "MARGIN_ONLY")


def test_analytic_margin_critical_tolerance_preserves_boundary_semantics() -> None:
    critical = analytic_margin_critical_tolerances(
        {"a": _point(3.0), "b": _point(0.0)},
        {"a": _point(0.0), "b": _point(3.0)},
        _config(1.0),
        minimum_scale=0.25,
        maximum_scale=4.0,
    )

    assert len(critical) == 1
    assert critical[0].scale == pytest.approx(3.0)
    assert critical[0].relation_below is Relation.STRUCTURAL_INCOMPARABLE
    assert critical[0].relation_at is Relation.STRUCTURAL_INCOMPARABLE
    assert critical[0].relation_above is Relation.UNRESOLVED


def test_transition_brackets_and_sampled_phases_are_explicit() -> None:
    spectrum = (
        SpectrumPoint(0.0, Relation.STRUCTURAL_INCOMPARABLE, ("a", "b")),
        SpectrumPoint(1.0, Relation.STRUCTURAL_INCOMPARABLE, ("a", "b")),
        SpectrumPoint(4.0, Relation.UNRESOLVED, ("a", "b")),
    )

    brackets = critical_tolerance_brackets(spectrum)
    assert len(brackets) == 1
    assert brackets[0].lower_scale == 1.0
    assert brackets[0].upper_scale == 4.0
    assert brackets[0].relation_before is Relation.STRUCTURAL_INCOMPARABLE
    assert brackets[0].relation_after is Relation.UNRESOLVED

    phases = sampled_relation_phases(spectrum)
    assert [(phase.start_scale, phase.end_scale, phase.relation) for phase in phases] == [
        (0.0, 1.0, Relation.STRUCTURAL_INCOMPARABLE),
        (4.0, 4.0, Relation.UNRESOLVED),
    ]
    assert [phase.sampled_point_count for phase in phases] == [2, 1]


def test_phase_diagram_and_cross_source_stability_are_identifier_free() -> None:
    source_a = aggregate_relation_phase_diagram(
        (
            (
                SpectrumPoint(0.5, Relation.STRUCTURAL_INCOMPARABLE, ("a", "b")),
                SpectrumPoint(1.0, Relation.UNRESOLVED, ("a", "b")),
            ),
            (
                SpectrumPoint(0.5, Relation.LESS, ("a",)),
                SpectrumPoint(1.0, Relation.LESS, ("a",)),
            ),
        )
    )
    source_b = aggregate_relation_phase_diagram(
        (
            (
                SpectrumPoint(0.5, Relation.STRUCTURAL_INCOMPARABLE, ("a", "b")),
                SpectrumPoint(1.0, Relation.STRUCTURAL_INCOMPARABLE, ("a", "b")),
            ),
            (
                SpectrumPoint(0.5, Relation.LESS, ("a",)),
                SpectrumPoint(1.0, Relation.UNRESOLVED, ("a",)),
            ),
        )
    )

    assert source_a[0].pair_count == 2
    assert source_a[0].proportions[Relation.STRUCTURAL_INCOMPARABLE] == 0.5
    stability = cross_source_stability(source_a, source_b)
    assert stability[0].agreement == 1.0
    assert stability[1].total_variation_distance == pytest.approx(0.5)
    assert stability[1].agreement == pytest.approx(0.5)
    integrated = integrated_cross_source_stability(stability)
    assert integrated.log2_scale_span == pytest.approx(1.0)
    assert integrated.normalized_integrated_total_variation == pytest.approx(0.25)
    assert integrated.integrated_agreement == pytest.approx(0.75)


def test_integrated_stability_rejects_nonpositive_or_short_grids() -> None:
    source = aggregate_relation_phase_diagram(
        ((SpectrumPoint(1.0, Relation.LESS, ("a",)),),)
    )
    one_point = cross_source_stability(source, source)
    with pytest.raises(ValueError, match="at least two"):
        integrated_cross_source_stability(one_point)

    invalid = (
        type(one_point[0])(
            scale=0.0,
            total_variation_distance=0.0,
            agreement=1.0,
            absolute_proportion_gaps=one_point[0].absolute_proportion_gaps,
        ),
        type(one_point[0])(
            scale=1.0,
            total_variation_distance=0.0,
            agreement=1.0,
            absolute_proportion_gaps=one_point[0].absolute_proportion_gaps,
        ),
    )
    with pytest.raises(ValueError, match="positive"):
        integrated_cross_source_stability(invalid)


@pytest.mark.parametrize("bad_scale", [-1.0, float("inf"), float("nan"), True])
def test_scale_validation_fails_closed(bad_scale: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ScaleRelationConfig(bad_scale, _config(0.0))


def test_scale_grid_and_source_grid_must_match_exactly() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate_relation_persistence(
            {"a": _point(1.0), "b": _point(0.0)},
            {"a": _point(0.0), "b": _point(1.0)},
            (
                ScaleRelationConfig(1.0, _config(0.0)),
                ScaleRelationConfig(1.0, _config(1.0)),
            ),
        )

    with pytest.raises(ValueError, match="common scale grid"):
        aggregate_relation_phase_diagram(
            (
                (SpectrumPoint(0.0, Relation.LESS, ("a",)),),
                (SpectrumPoint(1.0, Relation.LESS, ("a",)),),
            )
        )

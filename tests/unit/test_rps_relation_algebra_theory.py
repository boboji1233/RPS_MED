from __future__ import annotations

import random
from itertools import product

from nts_med.relations import Interval, Relation, RelationConfig, infer_relation
from rps_med import (
    ToleranceScalePolicy,
    build_scale_relation_configs,
    evaluate_relation_persistence,
)

_AXES = ("ma", "hem", "ex", "cws")
_SCALES = tuple(0.25 * 2.0 ** (quarter / 4.0) for quarter in range(17))
_SWAPPED = {
    Relation.LESS: Relation.GREATER,
    Relation.GREATER: Relation.LESS,
    Relation.EQUIVALENT: Relation.EQUIVALENT,
    Relation.STRUCTURAL_INCOMPARABLE: Relation.STRUCTURAL_INCOMPARABLE,
    Relation.UNRESOLVED: Relation.UNRESOLVED,
}


def _config(
    axes: tuple[str, ...],
    *,
    delta: tuple[float, ...],
    epsilon: tuple[float, ...],
    max_width: tuple[float, ...],
) -> RelationConfig:
    return RelationConfig(
        delta=dict(zip(axes, delta, strict=True)),
        epsilon=dict(zip(axes, epsilon, strict=True)),
        max_width=dict(zip(axes, max_width, strict=True)),
        numeric_tolerance=1.0e-12,
    )


def _compressed_relations(relations: tuple[Relation, ...]) -> tuple[Relation, ...]:
    output: list[Relation] = []
    for relation in relations:
        if not output or relation is not output[-1]:
            output.append(relation)
    return tuple(output)


def test_pair_swap_inversion_is_exhaustive_on_a_synthetic_interval_grid() -> None:
    axes = ("a", "b")
    atoms = (
        Interval(-1.0, -1.0),
        Interval(0.0, 0.0),
        Interval(0.0, 0.5),
        Interval(1.0, 2.0),
    )
    config = _config(
        axes,
        delta=(0.5, 1.0),
        epsilon=(0.25, 0.75),
        max_width=(0.5, 1.0),
    )

    for left_a, left_b, right_a, right_b in product(atoms, repeat=4):
        left = {"a": left_a, "b": left_b}
        right = {"a": right_a, "b": right_b}
        forward = infer_relation(left, right, config)
        reverse = infer_relation(right, left, config)
        assert reverse is _SWAPPED[forward]


def test_point_evidence_self_pairs_are_equivalent_at_every_frozen_scale() -> None:
    base = _config(
        _AXES,
        delta=(2.13210791, 7.33005532, 7.72235597, 8.15592432),
        epsilon=(0.50085046, 0.74138916, 0.19392044, 0.48575989),
        max_width=(0.0, 0.0, 0.0, 0.0),
    )
    configs = build_scale_relation_configs(
        base,
        _SCALES,
        ToleranceScalePolicy.MARGIN_ONLY,
    )

    for values in product((-2.0, 0.0, 3.5), repeat=len(_AXES)):
        evidence = {
            axis: Interval(value, value)
            for axis, value in zip(_AXES, values, strict=True)
        }
        spectrum = evaluate_relation_persistence(evidence, evidence, configs)
        assert all(point.relation is Relation.EQUIVALENT for point in spectrum)


def test_margin_only_analytic_spectra_have_at_most_one_transition_exhaustively() -> None:
    axes = ("a", "b")
    atoms = (
        Interval(-1.0, -1.0),
        Interval(0.0, 0.0),
        Interval(0.0, 0.5),
        Interval(1.0, 2.0),
    )
    base = _config(
        axes,
        delta=(0.5, 1.0),
        epsilon=(0.4, 0.4),
        max_width=(0.75, 0.75),
    )
    configs = build_scale_relation_configs(
        base,
        _SCALES,
        ToleranceScalePolicy.MARGIN_ONLY,
    )

    for left_a, left_b, right_a, right_b in product(atoms, repeat=4):
        spectrum = evaluate_relation_persistence(
            {"a": left_a, "b": left_b},
            {"a": right_a, "b": right_b},
            configs,
        )
        phases = _compressed_relations(tuple(point.relation for point in spectrum))
        assert len(phases) <= 2
        if len(phases) == 2:
            assert phases[0] in {
                Relation.LESS,
                Relation.GREATER,
                Relation.STRUCTURAL_INCOMPARABLE,
            }
            assert phases[1] in {Relation.EQUIVALENT, Relation.UNRESOLVED}


def test_margin_only_single_transition_property_on_seeded_synthetic_intervals() -> None:
    generator = random.Random(20260826)
    for _ in range(500):
        left: dict[str, Interval] = {}
        right: dict[str, Interval] = {}
        delta: list[float] = []
        epsilon: list[float] = []
        max_width: list[float] = []
        for axis in _AXES:
            left_lower = generator.uniform(-5.0, 5.0)
            right_lower = generator.uniform(-5.0, 5.0)
            left_width = generator.choice((0.0, 0.1, 0.5, 1.0))
            right_width = generator.choice((0.0, 0.1, 0.5, 1.0))
            left[axis] = Interval(left_lower, left_lower + left_width)
            right[axis] = Interval(right_lower, right_lower + right_width)
            delta.append(generator.uniform(0.1, 3.0))
            epsilon.append(generator.uniform(0.0, 1.0))
            max_width.append(generator.uniform(0.0, 1.0))

        base = _config(
            _AXES,
            delta=tuple(delta),
            epsilon=tuple(epsilon),
            max_width=tuple(max_width),
        )
        configs = build_scale_relation_configs(
            base,
            _SCALES,
            ToleranceScalePolicy.MARGIN_ONLY,
        )
        spectrum = evaluate_relation_persistence(left, right, configs)
        phases = _compressed_relations(tuple(point.relation for point in spectrum))
        assert len(phases) <= 2
        if len(phases) == 2:
            assert phases[0] in {
                Relation.LESS,
                Relation.GREATER,
                Relation.STRUCTURAL_INCOMPARABLE,
            }
            assert phases[1] in {Relation.EQUIVALENT, Relation.UNRESOLVED}

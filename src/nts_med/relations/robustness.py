"""Predeclared descriptive nulls and relation-stability summaries."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np

from nts_med.relations.algebra import Relation

BurdenTable = Mapping[str, Mapping[str, Real]]
RelationTable = Mapping[tuple[str, str], Relation]


@dataclass(frozen=True, slots=True)
class PairStability:
    """Sensitivity stability for one primary structurally incomparable pair."""

    pair: tuple[str, str]
    structural_incomparable_fraction: float


@dataclass(frozen=True, slots=True)
class RelationStabilitySummary:
    """Aggregate distribution of pairwise structural-relation stability."""

    pairs: tuple[PairStability, ...]
    median: float
    q25: float
    q75: float
    fraction_at_least_0_75: float
    variant_count: int


def _validated_burdens(burdens: BurdenTable) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not burdens:
        raise ValueError("burdens must not be empty")
    identifiers = tuple(sorted(burdens))
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError("instance identifiers must be non-empty strings")
    first_axes = tuple(sorted(burdens[identifiers[0]]))
    if not first_axes:
        raise ValueError("burdens must contain at least one axis")
    for identifier in identifiers:
        values = burdens[identifier]
        if tuple(sorted(values)) != first_axes:
            raise ValueError("all instances must use identical burden axes")
        for axis, value in values.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"burden {axis!r} must be a real number")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError("burdens must be finite and non-negative")
    return identifiers, first_axes


def comonotonic_burdens(burdens: BurdenTable) -> dict[str, dict[str, float]]:
    """Create the maximum-positive-dependence null while preserving marginals.

    Each axis is sorted independently and assigned to the same predeclared
    lexicographic opaque-instance order.  Every axis marginal is preserved
    exactly; original cross-axis case associations are deliberately removed.
    """

    identifiers, axes = _validated_burdens(burdens)
    result: dict[str, dict[str, float]] = {identifier: {} for identifier in identifiers}
    for axis in axes:
        ordered_values = sorted(float(burdens[identifier][axis]) for identifier in identifiers)
        for identifier, value in zip(identifiers, ordered_values, strict=True):
            result[identifier][axis] = value
    return result


def independently_permuted_burdens(
    burdens: BurdenTable, *, seed: int
) -> dict[str, dict[str, float]]:
    """Create a seeded independent-axis permutation null for description only."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    identifiers, axes = _validated_burdens(burdens)
    result: dict[str, dict[str, float]] = {identifier: {} for identifier in identifiers}
    rng = random.Random(seed)
    for axis in axes:
        permuted = [float(burdens[identifier][axis]) for identifier in identifiers]
        rng.shuffle(permuted)
        for identifier, value in zip(identifiers, permuted, strict=True):
            result[identifier][axis] = value
    return result


def structural_relation_stability(
    primary_relations: RelationTable,
    sensitivity_relations: Sequence[RelationTable],
) -> RelationStabilitySummary:
    """Summarize Sij across predeclared sensitivity relation tables."""

    if not sensitivity_relations:
        raise ValueError("at least one sensitivity relation table is required")
    primary_keys = set(primary_relations)
    if any(set(table) != primary_keys for table in sensitivity_relations):
        raise ValueError("all relation tables must contain identical pair keys")
    if any(
        not isinstance(relation, Relation)
        for table in (primary_relations, *sensitivity_relations)
        for relation in table.values()
    ):
        raise TypeError("relation table values must be Relation members")
    structural_pairs = tuple(
        sorted(
            pair
            for pair, relation in primary_relations.items()
            if relation is Relation.STRUCTURAL_INCOMPARABLE
        )
    )
    if not structural_pairs:
        raise ValueError("primary relation table contains no structural pairs")
    pair_stability = tuple(
        PairStability(
            pair=pair,
            structural_incomparable_fraction=sum(
                table[pair] is Relation.STRUCTURAL_INCOMPARABLE
                for table in sensitivity_relations
            )
            / len(sensitivity_relations),
        )
        for pair in structural_pairs
    )
    values = np.asarray(
        [item.structural_incomparable_fraction for item in pair_stability],
        dtype=np.float64,
    )
    return RelationStabilitySummary(
        pairs=pair_stability,
        median=float(np.median(values)),
        q25=float(np.quantile(values, 0.25, method="linear")),
        q75=float(np.quantile(values, 0.75, method="linear")),
        fraction_at_least_0_75=float(np.mean(values >= 0.75)),
        variant_count=len(sensitivity_relations),
    )


__all__ = [
    "PairStability",
    "RelationStabilitySummary",
    "comonotonic_burdens",
    "independently_permuted_burdens",
    "structural_relation_stability",
]

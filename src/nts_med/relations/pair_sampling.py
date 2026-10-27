"""Small deterministic utilities for sampling directed instance pairs."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, order=True, slots=True)
class DirectedPair:
    """A directed, non-reflexive pair of instance identifiers."""

    left_id: str
    right_id: str

    def __post_init__(self) -> None:
        if not self.left_id or not self.right_id:
            raise ValueError("pair identifiers must be non-empty")
        if self.left_id == self.right_id:
            raise ValueError("a directed pair cannot contain the same instance twice")


@dataclass(frozen=True, order=True, slots=True)
class UnorderedPair:
    """A canonical, non-reflexive unordered pair of instance identifiers."""

    first_id: str
    second_id: str

    def __post_init__(self) -> None:
        if not self.first_id or not self.second_id:
            raise ValueError("pair identifiers must be non-empty")
        if self.first_id >= self.second_id:
            raise ValueError("unordered pair identifiers must be strictly sorted")


def all_unordered_pairs(instance_ids: Sequence[str]) -> tuple[UnorderedPair, ...]:
    """Enumerate every legal unordered instance pair deterministically."""

    if isinstance(instance_ids, (str, bytes)):
        raise TypeError("instance_ids must be a sequence of strings")
    identifiers = tuple(sorted(instance_ids))
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError("instance identifiers must be non-empty strings")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("instance identifiers must be unique")
    return tuple(
        UnorderedPair(identifiers[left], identifiers[right])
        for left in range(len(identifiers))
        for right in range(left + 1, len(identifiers))
    )


def sample_directed_pairs(
    instance_ids: Sequence[str],
    *,
    pairs_per_anchor: int,
    seed: int,
) -> tuple[DirectedPair, ...]:
    """Sample a fixed number of partners per anchor reproducibly.

    Identifiers are sorted before sampling, so permutation of the caller's
    input does not change the resulting manifest.
    """

    if isinstance(instance_ids, (str, bytes)):
        raise TypeError("instance_ids must be a sequence of strings")
    if isinstance(pairs_per_anchor, bool) or not isinstance(pairs_per_anchor, int):
        raise TypeError("pairs_per_anchor must be an integer")
    if pairs_per_anchor < 0:
        raise ValueError("pairs_per_anchor must be non-negative")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    identifiers = tuple(sorted(instance_ids))
    if any(
        not isinstance(identifier, str) or not identifier for identifier in identifiers
    ):
        raise ValueError("instance identifiers must be non-empty strings")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("instance identifiers must be unique")
    if identifiers and pairs_per_anchor > len(identifiers) - 1:
        raise ValueError(
            "pairs_per_anchor cannot exceed the number of available partners"
        )
    if not identifiers and pairs_per_anchor:
        raise ValueError("cannot sample pairs from an empty identifier sequence")

    random_generator = random.Random(seed)
    sampled: list[DirectedPair] = []
    for anchor in identifiers:
        candidates = [identifier for identifier in identifiers if identifier != anchor]
        partners = random_generator.sample(candidates, pairs_per_anchor)
        sampled.extend(DirectedPair(anchor, partner) for partner in sorted(partners))
    return tuple(sampled)


__all__ = [
    "DirectedPair",
    "UnorderedPair",
    "all_unordered_pairs",
    "sample_directed_pairs",
]

"""Deterministic cluster-bootstrap interfaces for later Gate 0 execution."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from nts_med.relations.algebra import Relation
from nts_med.relations.pair_sampling import all_unordered_pairs


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    """Frozen bootstrap controls; this object does not compute an endpoint."""

    replicates: int = 5000
    seed: int = 20260823
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if isinstance(self.replicates, bool) or not isinstance(self.replicates, Integral):
            raise TypeError("replicates must be an integer")
        if int(self.replicates) < 1:
            raise ValueError("replicates must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")
        if isinstance(self.confidence_level, bool) or not isinstance(
            self.confidence_level, Real
        ):
            raise TypeError("confidence_level must be a real number")
        confidence = float(self.confidence_level)
        if not math.isfinite(confidence) or not 0 < confidence < 1:
            raise ValueError("confidence_level must be finite and in (0, 1)")


@dataclass(frozen=True, slots=True)
class ProportionInterval:
    """Percentile interval for one relation proportion."""

    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class ImageResamplingBootstrapResult:
    """Image-resampling U-statistic summary, never a pair-row bootstrap."""

    point_proportions: Mapping[Relation, float]
    confidence_intervals: Mapping[Relation, ProportionInterval]
    valid_replicates: int
    degenerate_replicates: int
    requested_replicates: int
    seed: int
    confidence_level: float


def _validated_strata(
    strata: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not strata:
        raise ValueError("strata must not be empty")
    normalized: list[tuple[str, tuple[str, ...]]] = []
    seen_clusters: set[str] = set()
    for stratum in sorted(strata):
        if not isinstance(stratum, str) or not stratum:
            raise ValueError("stratum names must be non-empty strings")
        clusters = tuple(strata[stratum])
        if not clusters:
            raise ValueError("each stratum must contain at least one cluster")
        if any(not isinstance(cluster, str) or not cluster for cluster in clusters):
            raise ValueError("cluster identifiers must be non-empty strings")
        if len(set(clusters)) != len(clusters):
            raise ValueError("cluster identifiers must be unique within each stratum")
        if seen_clusters & set(clusters):
            raise ValueError("cluster identifiers must not occur in multiple strata")
        seen_clusters.update(clusters)
        normalized.append((stratum, tuple(sorted(clusters))))
    return tuple(normalized)


def stratified_cluster_resamples(
    strata: Mapping[str, Sequence[str]],
    config: BootstrapConfig | None = None,
) -> tuple[dict[str, tuple[str, ...]], ...]:
    """Draw whole clusters with replacement, independently within each stratum."""

    resolved = config or BootstrapConfig()
    normalized = _validated_strata(strata)
    rng = random.Random(int(resolved.seed))
    resamples: list[dict[str, tuple[str, ...]]] = []
    for _ in range(int(resolved.replicates)):
        replicate: dict[str, tuple[str, ...]] = {}
        for stratum, clusters in normalized:
            replicate[stratum] = tuple(
                clusters[rng.randrange(len(clusters))] for _ in clusters
            )
        resamples.append(replicate)
    return tuple(resamples)


def tail_probability_monte_carlo_se(
    replicates: int, *, tail_probability: float = 0.025
) -> float:
    """Return binomial Monte Carlo SE for a requested tail probability."""

    if isinstance(replicates, bool) or not isinstance(replicates, Integral):
        raise TypeError("replicates must be an integer")
    count = int(replicates)
    if count < 1:
        raise ValueError("replicates must be positive")
    if isinstance(tail_probability, bool) or not isinstance(tail_probability, Real):
        raise TypeError("tail_probability must be a real number")
    probability = float(tail_probability)
    if not math.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("tail_probability must be finite and in (0, 1)")
    return math.sqrt(probability * (1.0 - probability) / count)


def resampled_unordered_pair_weights(
    instance_ids: Sequence[str],
    sampled_instance_ids: Sequence[str],
) -> dict[tuple[str, str], int]:
    """Return multiplicity weights for legal pairs in one image resample.

    Multiple sampled copies of the same source instance do not form a legal
    pair.  For two different source instances, the pair weight is the product
    of their sampled multiplicities.
    """

    legal = tuple(pair for pair in all_unordered_pairs(instance_ids))
    identifiers = {identifier for pair in legal for identifier in (pair.first_id, pair.second_id)}
    if len(instance_ids) == 1:
        only = tuple(instance_ids)
        if len(only) != 1 or not isinstance(only[0], str) or not only[0]:
            raise ValueError("instance identifiers must be non-empty strings")
        identifiers = {only[0]}
    if isinstance(sampled_instance_ids, (str, bytes)):
        raise TypeError("sampled_instance_ids must be a sequence of strings")
    sampled = tuple(sampled_instance_ids)
    if any(identifier not in identifiers for identifier in sampled):
        raise ValueError("sampled instance identifiers must belong to instance_ids")
    multiplicities = Counter(sampled)
    return {
        (pair.first_id, pair.second_id): (
            multiplicities[pair.first_id] * multiplicities[pair.second_id]
        )
        for pair in legal
        if multiplicities[pair.first_id] and multiplicities[pair.second_id]
    }


def image_resampling_relation_bootstrap(
    instance_ids: Sequence[str],
    relation_lookup: Mapping[tuple[str, str], Relation],
    config: BootstrapConfig | None = None,
) -> ImageResamplingBootstrapResult:
    """Bootstrap relation proportions by resampling images, then recomputing pairs."""

    resolved = config or BootstrapConfig()
    pairs = all_unordered_pairs(instance_ids)
    if not pairs:
        raise ValueError("at least two unique instances are required")
    expected_keys = {(pair.first_id, pair.second_id) for pair in pairs}
    if set(relation_lookup) != expected_keys:
        missing = len(expected_keys - set(relation_lookup))
        extra = len(set(relation_lookup) - expected_keys)
        raise ValueError(
            "relation_lookup must cover all unordered pairs; "
            f"missing={missing}, extra={extra}"
        )
    if any(not isinstance(value, Relation) for value in relation_lookup.values()):
        raise TypeError("relation_lookup values must be Relation members")

    relations = tuple(Relation)
    point_counts = Counter(relation_lookup.values())
    point = {relation: point_counts[relation] / len(pairs) for relation in relations}
    estimates: dict[Relation, list[float]] = {relation: [] for relation in relations}
    identifiers = tuple(sorted(instance_ids))
    rng = random.Random(int(resolved.seed))
    degenerate = 0
    for _ in range(int(resolved.replicates)):
        sampled = tuple(identifiers[rng.randrange(len(identifiers))] for _ in identifiers)
        weights = resampled_unordered_pair_weights(identifiers, sampled)
        total_weight = sum(weights.values())
        if total_weight == 0:
            degenerate += 1
            continue
        counts: Counter[Relation] = Counter()
        for pair, weight in weights.items():
            counts[relation_lookup[pair]] += weight
        for relation in relations:
            estimates[relation].append(counts[relation] / total_weight)
    valid = int(resolved.replicates) - degenerate
    if valid == 0:
        raise ValueError("all bootstrap replicates were degenerate")
    alpha = (1.0 - float(resolved.confidence_level)) / 2.0
    intervals = {
        relation: ProportionInterval(
            lower=float(np.quantile(estimates[relation], alpha, method="linear")),
            upper=float(np.quantile(estimates[relation], 1.0 - alpha, method="linear")),
        )
        for relation in relations
    }
    return ImageResamplingBootstrapResult(
        point_proportions=point,
        confidence_intervals=intervals,
        valid_replicates=valid,
        degenerate_replicates=degenerate,
        requested_replicates=int(resolved.replicates),
        seed=int(resolved.seed),
        confidence_level=float(resolved.confidence_level),
    )


__all__ = [
    "BootstrapConfig", "ImageResamplingBootstrapResult", "ProportionInterval",
    "image_resampling_relation_bootstrap", "resampled_unordered_pair_weights",
    "stratified_cluster_resamples",
    "tail_probability_monte_carlo_se",
]

"""Path-free frozen-feature similarity for duplicate-candidate ranking.

Feature similarity is an audit signal, not proof that two medical images share
an acquisition or patient.  This module deliberately contains no model loading,
network access, file deletion, or split mutation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CrossSplitFeatureCandidate:
    """One ranked cross-split pair selected by threshold and/or top-k."""

    left_id: str
    right_id: str
    left_split: str
    right_split: str
    cosine_similarity: float
    selected_by_threshold: bool
    selected_by_top_k: bool

    def __post_init__(self) -> None:
        if self.left_id >= self.right_id:
            raise ValueError("candidate identifiers must be in strict sorted order")
        if self.left_split == self.right_split:
            raise ValueError("feature candidates must span different splits")
        if not math.isfinite(self.cosine_similarity):
            raise ValueError("cosine_similarity must be finite")
        if not -1.0000001 <= self.cosine_similarity <= 1.0000001:
            raise ValueError("cosine_similarity must lie in [-1, 1]")
        if not self.selected_by_threshold and not self.selected_by_top_k:
            raise ValueError("a candidate needs at least one selection reason")


def _require_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("minimum_similarity must be a real number")
    result = float(value)
    if not math.isfinite(result) or not -1.0 <= result <= 1.0:
        raise ValueError("minimum_similarity must be finite and lie in [-1, 1]")
    return result


def _require_top_k(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("top_k must be an integer")
    if value < 0:
        raise ValueError("top_k must be non-negative")
    return value


def _normalized_feature_matrix(
    features: Mapping[str, NDArray[np.floating[Any]]],
    identifiers: tuple[str, ...],
) -> NDArray[np.float64]:
    if not identifiers:
        return np.empty((0, 0), dtype=np.float64)
    arrays: list[NDArray[np.float64]] = []
    dimension: int | None = None
    for instance_id in identifiers:
        array = np.asarray(features[instance_id], dtype=np.float64)
        if array.ndim != 1 or array.size == 0:
            raise ValueError("each feature vector must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(array)):
            raise ValueError("feature vectors must contain only finite values")
        if dimension is None:
            dimension = int(array.size)
        elif array.size != dimension:
            raise ValueError("feature vectors must have a consistent dimension")
        norm = float(np.linalg.norm(array))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("feature vectors must have a positive finite norm")
        arrays.append(array / norm)
    return np.vstack(arrays)


def find_cross_split_feature_candidates(
    features: Mapping[str, NDArray[np.floating[Any]]],
    split_by_instance: Mapping[str, str],
    *,
    minimum_similarity: float,
    top_k: int,
) -> tuple[CrossSplitFeatureCandidate, ...]:
    """Rank cross-split pairs and return the threshold/top-k union.

    The result is deterministic: identifiers are canonicalized before pair
    construction, and ties are broken lexicographically.  Selecting a pair does
    not imply that it is a duplicate; every returned row remains review-only.
    """

    threshold = _require_threshold(minimum_similarity)
    requested_top_k = _require_top_k(top_k)
    identifiers = tuple(sorted(features))
    if set(split_by_instance) != set(identifiers):
        raise ValueError("split mapping must exactly cover feature instances")
    for instance_id, split in split_by_instance.items():
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("feature identifiers must be non-empty strings")
        if not isinstance(split, str) or not split:
            raise ValueError(f"split for {instance_id!r} must be a non-empty string")

    matrix = _normalized_feature_matrix(features, identifiers)
    ranked_pairs: list[tuple[float, str, str, str, str]] = []
    for left_index, left_id in enumerate(identifiers):
        for right_index in range(left_index + 1, len(identifiers)):
            right_id = identifiers[right_index]
            left_split = split_by_instance[left_id]
            right_split = split_by_instance[right_id]
            if left_split == right_split:
                continue
            similarity = float(np.clip(matrix[left_index] @ matrix[right_index], -1.0, 1.0))
            ranked_pairs.append(
                (similarity, left_id, right_id, left_split, right_split)
            )

    ranked_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    top_keys = {
        (left_id, right_id)
        for _, left_id, right_id, _, _ in ranked_pairs[:requested_top_k]
    }
    candidates = [
        CrossSplitFeatureCandidate(
            left_id=left_id,
            right_id=right_id,
            left_split=left_split,
            right_split=right_split,
            cosine_similarity=similarity,
            selected_by_threshold=similarity >= threshold,
            selected_by_top_k=(left_id, right_id) in top_keys,
        )
        for similarity, left_id, right_id, left_split, right_split in ranked_pairs
        if similarity >= threshold or (left_id, right_id) in top_keys
    ]
    return tuple(candidates)


__all__ = [
    "CrossSplitFeatureCandidate",
    "find_cross_split_feature_candidates",
]

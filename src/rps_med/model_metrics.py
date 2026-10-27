"""Pure target construction and metrics for RPS-Med model experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nts_med.gate0.execution import MeasurementRecord
from nts_med.relations import Relation, RelationConfig
from rps_med.formal import build_pair_feature_table


@dataclass(frozen=True, slots=True)
class PairTargets:
    left_indices: NDArray[np.int64]
    right_indices: NDArray[np.int64]
    relation_codes: NDArray[np.int64]
    transition: NDArray[np.float32]
    critical_log2: NDArray[np.float32]

    @property
    def pair_count(self) -> int:
        return int(self.left_indices.size)


def build_relation_targets(
    burdens: NDArray[np.float32] | NDArray[np.float64],
    indices: Sequence[int],
    *,
    base_config: RelationConfig,
    scales: Sequence[float],
) -> PairTargets:
    """Build oriented relation labels for all unordered pairs in one partition."""

    selected = np.asarray(tuple(indices), dtype=np.int64)
    if burdens.ndim != 2 or burdens.shape[1] != 4:
        raise ValueError("burdens must have shape [instances, 4]")
    if selected.size < 2 or bool((selected < 0).any()) or bool(
        (selected >= burdens.shape[0]).any()
    ):
        raise ValueError("pair target indices are invalid")
    axes = ("ma", "hem", "ex", "cws")
    records = tuple(
        MeasurementRecord(
            instance_id=f"model_{local_index:06d}",
            primary={
                axis: float(burdens[index, axis_index])
                for axis_index, axis in enumerate(axes)
            },
            variants={},
            grade=None,
            grading_split_conflict=False,
            strict_measurement_quality=True,
        )
        for local_index, index in enumerate(selected)
    )
    table = build_pair_feature_table(records, base_config=base_config, fine_scales=scales)
    critical = np.asarray(table.critical_scale, dtype=np.float64)
    log_critical = np.full(critical.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(critical)
    log_critical[finite] = np.asarray(np.log2(critical[finite]), dtype=np.float32)
    return PairTargets(
        left_indices=selected[np.asarray(table.left_indices, dtype=np.int64)],
        right_indices=selected[np.asarray(table.right_indices, dtype=np.int64)],
        relation_codes=np.asarray(table.fine_relation_codes, dtype=np.int64),
        transition=np.asarray(table.transition, dtype=np.float32),
        critical_log2=log_critical,
    )


def multiscale_macro_f1(
    truth: NDArray[np.int64], prediction: NDArray[np.int64], *, class_count: int = 5
) -> float:
    """Average per-scale macro F1 over classes present in ground truth."""

    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError("relation arrays must have identical [pair, scale] shape")
    scale_scores: list[float] = []
    for scale_index in range(truth.shape[1]):
        class_scores: list[float] = []
        for code in range(class_count):
            actual = truth[:, scale_index] == code
            if not bool(actual.any()):
                continue
            predicted = prediction[:, scale_index] == code
            true_positive = int(np.count_nonzero(np.logical_and(actual, predicted)))
            false_positive = int(np.count_nonzero(np.logical_and(~actual, predicted)))
            false_negative = int(np.count_nonzero(np.logical_and(actual, ~predicted)))
            denominator = 2 * true_positive + false_positive + false_negative
            class_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
        if not class_scores:
            raise ValueError("each scale must have at least one supported class")
        scale_scores.append(float(np.mean(class_scores)))
    return float(np.mean(scale_scores))


def relation_accuracy(
    truth: NDArray[np.int64], prediction: NDArray[np.int64]
) -> float:
    if truth.shape != prediction.shape or not truth.size:
        raise ValueError("relation arrays must be non-empty with identical shape")
    return float(np.mean(truth == prediction))


def phase_total_variation(
    truth: NDArray[np.int64], prediction: NDArray[np.int64], *, class_count: int = 5
) -> float:
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError("relation arrays must have identical [pair, scale] shape")
    values = []
    for scale_index in range(truth.shape[1]):
        truth_counts = np.bincount(truth[:, scale_index], minlength=class_count) / truth.shape[0]
        pred_counts = (
            np.bincount(prediction[:, scale_index], minlength=class_count) / truth.shape[0]
        )
        values.append(0.5 * float(np.abs(truth_counts - pred_counts).sum()))
    return float(np.mean(values))


def binary_auc(truth: NDArray[np.float32], score: NDArray[np.float32]) -> float | None:
    """Return rank-based ROC AUC, or None when either class is absent."""

    if truth.shape != score.shape or truth.ndim != 1:
        raise ValueError("binary arrays must have identical one-dimensional shape")
    positive = truth > 0.5
    positive_count = int(np.count_nonzero(positive))
    negative_count = int(truth.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(score, kind="stable")
    ranks = np.empty(score.size, dtype=np.float64)
    position = 0
    while position < score.size:
        end = position + 1
        while end < score.size and score[order[end]] == score[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + 1 + end) / 2.0
        position = end
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def critical_log2_mae(
    truth: NDArray[np.float32], prediction: NDArray[np.float32]
) -> float | None:
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("critical arrays must have identical one-dimensional shape")
    finite = np.isfinite(truth)
    if not bool(finite.any()):
        return None
    value = float(np.mean(np.abs(truth[finite] - prediction[finite])))
    return value if math.isfinite(value) else None


def majority_spectrum_baseline(train: PairTargets, target_pair_count: int) -> NDArray[np.int64]:
    """Predict each scale's most frequent training relation for every target pair."""

    if target_pair_count <= 0:
        raise ValueError("target pair count must be positive")
    modes = np.asarray(
        [
            np.bincount(train.relation_codes[:, index], minlength=len(Relation)).argmax()
            for index in range(train.relation_codes.shape[1])
        ],
        dtype=np.int64,
    )
    return np.tile(modes, (target_pair_count, 1))


__all__ = [
    "PairTargets",
    "binary_auc",
    "build_relation_targets",
    "critical_log2_mae",
    "majority_spectrum_baseline",
    "multiscale_macro_f1",
    "phase_total_variation",
    "relation_accuracy",
]

"""Development-only model heads and diagnostics for RPS-Med Stage 3A."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from nts_med.relations import Relation
from rps_med.model_metrics import (
    PairTargets,
    binary_auc,
    critical_log2_mae,
    multiscale_macro_f1,
    phase_total_variation,
    relation_accuracy,
)

HeadKind = Literal[
    "joint_spectrum",
    "independent_scale",
    "lambda_conditioned_shared",
    "merged_si_unresolved_four_state",
]
PairRepresentation = Literal["full", "signed_difference", "absolute_difference"]

_INVERSION = np.asarray((1, 0, 2, 3, 4), dtype=np.int64)
_ADMISSIBLE_TRANSITIONS = frozenset(
    {
        (0, 2),
        (0, 4),
        (1, 2),
        (1, 4),
        (3, 2),
        (3, 4),
    }
)


def pair_features(
    left: Tensor, right: Tensor, representation: PairRepresentation
) -> Tensor:
    """Return the preregistered orientation-aware pair representation."""

    signed = left - right
    if representation == "signed_difference":
        return signed
    absolute = torch.abs(signed)
    if representation == "absolute_difference":
        return absolute
    if representation == "full":
        return torch.cat((signed, absolute, left * right), dim=1)
    raise ValueError("unknown Stage 3A pair representation")


class ConfigurablePairHead(nn.Module):
    """Frozen-budget Stage 3A spectrum head with optional auxiliary outputs."""

    normalized_log2_scales: Tensor

    def __init__(
        self,
        *,
        head_kind: HeadKind,
        pair_representation: PairRepresentation = "full",
        embedding_dim: int = 512,
        scale_count: int = 17,
        relation_count: int = 5,
        dropout: float = 0.20,
        transition_output: bool = False,
        critical_output: bool = False,
        scales: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.head_kind = head_kind
        self.pair_representation = pair_representation
        self.scale_count = scale_count
        self.relation_count = relation_count
        self.has_transition_output = transition_output
        self.has_critical_output = critical_output
        feature_multiplier = 3 if pair_representation == "full" else 1
        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim * feature_multiplier, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if head_kind in {"joint_spectrum", "merged_si_unresolved_four_state"}:
            self.spectrum: nn.Module = nn.Linear(256, scale_count * relation_count)
        elif head_kind == "independent_scale":
            self.spectrum = nn.ModuleList(
                nn.Linear(256, relation_count) for _ in range(scale_count)
            )
        elif head_kind == "lambda_conditioned_shared":
            self.spectrum = nn.Linear(257, relation_count)
        else:
            raise ValueError("unknown Stage 3A head kind")
        if scales is None:
            scales = tuple(2.0 ** (-2.0 + index * 0.25) for index in range(scale_count))
        if len(scales) != scale_count:
            raise ValueError("scale values do not match the Stage 3A head")
        normalized_log2 = torch.tensor(
            [math.log2(value) / 2.0 for value in scales], dtype=torch.float32
        )
        self.register_buffer("normalized_log2_scales", normalized_log2)
        self.transition = nn.Linear(256, 1) if transition_output else None
        self.critical_log2 = nn.Linear(256, 1) if critical_output else None

    def forward(self, left: Tensor, right: Tensor) -> tuple[Tensor, Tensor | None, Tensor | None]:
        hidden = self.trunk(pair_features(left, right, self.pair_representation))
        if self.head_kind in {"joint_spectrum", "merged_si_unresolved_four_state"}:
            logits = self.spectrum(hidden).reshape(
                -1, self.scale_count, self.relation_count
            )
        elif self.head_kind == "independent_scale":
            assert isinstance(self.spectrum, nn.ModuleList)
            logits = torch.stack([head(hidden) for head in self.spectrum], dim=1)
        else:
            assert isinstance(self.spectrum, nn.Linear)
            expanded = hidden[:, None, :].expand(-1, self.scale_count, -1)
            scale = self.normalized_log2_scales[None, :, None].expand(
                hidden.shape[0], -1, -1
            )
            logits = self.spectrum(torch.cat((expanded, scale), dim=2))
        transition = None if self.transition is None else self.transition(hidden).squeeze(1)
        critical = (
            None if self.critical_log2 is None else self.critical_log2(hidden).squeeze(1)
        )
        return logits, transition, critical


class RankNetScalarHead(nn.Module):
    """Scalar-order stress test that can only emit LESS/GREATER/EQUIVALENT."""

    def __init__(self, *, embedding_dim: int = 512, scale_count: int = 17) -> None:
        super().__init__()
        self.scale_count = scale_count
        self.score = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )
        self.raw_cutpoints = nn.Parameter(torch.zeros(scale_count))

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        difference = self.score(left).squeeze(1) - self.score(right).squeeze(1)
        cutpoints = torch.nn.functional.softplus(self.raw_cutpoints)[None, :]
        expanded = difference[:, None]
        return torch.stack(
            (
                -expanded - cutpoints,
                expanded - cutpoints,
                cutpoints - torch.abs(expanded),
            ),
            dim=2,
        )


def soft_admissibility_penalty(logits: Tensor) -> Tensor:
    """Penalize probabilistic adjacent-state variation beyond one transition."""

    probabilities = torch.softmax(logits, dim=2)
    adjacent_tv = 0.5 * torch.abs(probabilities[:, 1:] - probabilities[:, :-1]).sum(dim=2)
    excess = torch.relu(adjacent_tv.sum(dim=1) - 1.0)
    return torch.mean(excess.square())


def pooled_per_state_f1(
    truth: NDArray[np.int64], prediction: NDArray[np.int64], *, class_count: int = 5
) -> dict[str, float | None]:
    """Compute the frozen pooled pair-by-scale F1 for each relation state."""

    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError("per-state arrays must align")
    names = [relation.value for relation in Relation]
    if class_count == 4:
        names = [*names[:3], "MERGED_SI_UNRESOLVED"]
    output: dict[str, float | None] = {}
    flat_truth = truth.ravel()
    flat_prediction = prediction.ravel()
    for code, name in enumerate(names[:class_count]):
        actual = flat_truth == code
        if not bool(actual.any()):
            output[name] = None
            continue
        predicted = flat_prediction == code
        true_positive = int(np.count_nonzero(np.logical_and(actual, predicted)))
        false_positive = int(np.count_nonzero(np.logical_and(~actual, predicted)))
        false_negative = int(np.count_nonzero(np.logical_and(actual, ~predicted)))
        denominator = 2 * true_positive + false_positive + false_negative
        output[name] = 0.0 if denominator == 0 else 2.0 * true_positive / denominator
    return output


def path_validity_rates(prediction: NDArray[np.int64]) -> tuple[float, float]:
    """Return invalid-family and multi-transition rates for five-state paths."""

    if prediction.ndim != 2 or prediction.shape[1] != 17:
        raise ValueError("Stage 3A path diagnostics require [pair, 17]")
    invalid = 0
    multi = 0
    for row in prediction:
        compressed = [int(row[0])]
        for value in row[1:]:
            code = int(value)
            if code != compressed[-1]:
                compressed.append(code)
        if len(compressed) > 2:
            multi += 1
            invalid += 1
        elif len(compressed) == 2 and tuple(compressed) not in _ADMISSIBLE_TRANSITIONS:
            invalid += 1
    count = prediction.shape[0]
    return invalid / count, multi / count


def pair_swap_consistency(
    forward: NDArray[np.int64], reverse: NDArray[np.int64], *, class_count: int = 5
) -> float:
    """Compare forward predictions with the converse of reversed predictions."""

    if forward.shape != reverse.shape:
        raise ValueError("pair-swap predictions must align")
    inversion = _INVERSION if class_count == 5 else np.asarray((1, 0, 2, 3), dtype=np.int64)
    return float(np.mean(forward == inversion[reverse]))


@dataclass(frozen=True, slots=True)
class Stage3AMetrics:
    spectrum_macro_f1: float
    relation_accuracy: float
    phase_total_variation: float
    per_state_f1: dict[str, float | None]
    transition_auc: float | None
    critical_log2_mae: float | None
    pair_swap_consistency: float
    invalid_spectrum_rate: float | None
    multi_transition_rate: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_stage3a_predictions(
    truth: PairTargets,
    prediction: NDArray[np.int64],
    reverse_prediction: NDArray[np.int64],
    *,
    transition_score: NDArray[np.float32] | None = None,
    critical_prediction: NDArray[np.float32] | None = None,
    class_count: int = 5,
) -> Stage3AMetrics:
    """Evaluate one frozen validation prediction set without resampling pairs."""

    relation_truth = truth.relation_codes
    if class_count == 4:
        relation_truth = relation_truth.copy()
        relation_truth[relation_truth == 4] = 3
    invalid: float | None = None
    multi: float | None = None
    if class_count == 5:
        invalid, multi = path_validity_rates(prediction)
    return Stage3AMetrics(
        spectrum_macro_f1=multiscale_macro_f1(
            relation_truth, prediction, class_count=class_count
        ),
        relation_accuracy=relation_accuracy(relation_truth, prediction),
        phase_total_variation=phase_total_variation(
            relation_truth, prediction, class_count=class_count
        ),
        per_state_f1=pooled_per_state_f1(
            relation_truth, prediction, class_count=class_count
        ),
        transition_auc=(
            None if transition_score is None else binary_auc(truth.transition, transition_score)
        ),
        critical_log2_mae=(
            None
            if critical_prediction is None
            else critical_log2_mae(truth.critical_log2, critical_prediction)
        ),
        pair_swap_consistency=pair_swap_consistency(
            prediction, reverse_prediction, class_count=class_count
        ),
        invalid_spectrum_rate=invalid,
        multi_transition_rate=multi,
    )


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


__all__ = [
    "ConfigurablePairHead",
    "HeadKind",
    "PairRepresentation",
    "RankNetScalarHead",
    "Stage3AMetrics",
    "evaluate_stage3a_predictions",
    "pair_features",
    "pair_swap_consistency",
    "parameter_count",
    "path_validity_rates",
    "pooled_per_state_f1",
    "soft_admissibility_penalty",
]

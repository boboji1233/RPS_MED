from __future__ import annotations

import numpy as np
import torch

from rps_med.stage3a_models import (
    ConfigurablePairHead,
    RankNetScalarHead,
    pair_features,
    pair_swap_consistency,
    path_validity_rates,
    pooled_per_state_f1,
    soft_admissibility_penalty,
)


def test_stage3a_pair_representations_have_frozen_dimensions() -> None:
    left = torch.tensor([[1.0, 2.0]])
    right = torch.tensor([[0.5, 3.0]])
    signed = pair_features(left, right, "signed_difference")
    absolute = pair_features(left, right, "absolute_difference")
    full = pair_features(left, right, "full")
    assert signed.tolist() == [[0.5, -1.0]]
    assert absolute.tolist() == [[0.5, 1.0]]
    assert full.shape == (1, 6)
    assert full[:, -2:].tolist() == [[0.5, 6.0]]


def test_stage3a_heads_emit_expected_contract_shapes() -> None:
    left = torch.randn(3, 8)
    right = torch.randn(3, 8)
    for kind in ("joint_spectrum", "independent_scale", "lambda_conditioned_shared"):
        head = ConfigurablePairHead(
            head_kind=kind,
            embedding_dim=8,
            transition_output=True,
            critical_output=True,
        )
        spectrum, transition, critical = head(left, right)
        assert spectrum.shape == (3, 17, 5)
        assert transition is not None and transition.shape == (3,)
        assert critical is not None and critical.shape == (3,)

    merged = ConfigurablePairHead(
        head_kind="merged_si_unresolved_four_state",
        embedding_dim=8,
        relation_count=4,
    )
    spectrum, transition, critical = merged(left, right)
    assert spectrum.shape == (3, 17, 4)
    assert transition is None and critical is None


def test_ranknet_stress_test_cannot_emit_si_or_unresolved() -> None:
    head = RankNetScalarHead(embedding_dim=8)
    logits = head(torch.randn(4, 8), torch.randn(4, 8))
    assert logits.shape == (4, 17, 3)
    assert int(logits.argmax(dim=2).max()) <= 2


def test_admissibility_penalty_distinguishes_stable_and_oscillating_probabilities() -> None:
    stable = torch.full((2, 17, 5), -20.0)
    stable[:, :, 4] = 20.0
    oscillating = torch.full((2, 17, 5), -20.0)
    for index in range(17):
        oscillating[:, index, index % 2] = 20.0
    assert float(soft_admissibility_penalty(stable)) == 0.0
    assert float(soft_admissibility_penalty(oscillating)) > 1.0


def test_path_validity_uses_the_preregistered_analytic_family() -> None:
    valid_constant = np.full(17, 4, dtype=np.int64)
    valid_transition = np.asarray([3] * 8 + [4] * 9, dtype=np.int64)
    invalid_direction = np.asarray([2] * 8 + [3] * 9, dtype=np.int64)
    multi = np.asarray([0] * 5 + [4] * 5 + [2] * 7, dtype=np.int64)
    invalid_rate, multi_rate = path_validity_rates(
        np.stack((valid_constant, valid_transition, invalid_direction, multi))
    )
    assert invalid_rate == 0.5
    assert multi_rate == 0.25


def test_pair_swap_and_pooled_state_f1_are_orientation_aware() -> None:
    forward = np.asarray([[0, 1, 2, 3, 4]], dtype=np.int64)
    reverse = np.asarray([[1, 0, 2, 3, 4]], dtype=np.int64)
    assert pair_swap_consistency(forward, reverse) == 1.0
    scores = pooled_per_state_f1(forward, forward)
    assert all(value == 1.0 for value in scores.values())

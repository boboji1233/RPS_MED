from __future__ import annotations

import numpy as np
import pytest
import torch

from rps_med.stage3a_models import path_validity_rates
from rps_med.stage3b_grammar import (
    canonical_path_ids,
    canonical_paths,
    grammar_decode,
    minimum_hamming_projection,
    path_scores,
    structured_path_nll,
)


def test_canonical_grammar_contains_exactly_101_valid_unique_paths() -> None:
    paths = canonical_paths()
    assert paths.shape == (101, 17)
    assert np.unique(paths, axis=0).shape == paths.shape
    invalid, multi = path_validity_rates(paths)
    assert invalid == 0.0
    assert multi == 0.0
    assert np.array_equal(canonical_path_ids(paths), np.arange(101))


def test_canonical_path_ids_reject_out_of_grammar_truth() -> None:
    invalid = np.zeros((1, 17), dtype=np.int64)
    invalid[0, 8:] = 1
    with pytest.raises(ValueError, match="outside the frozen grammar"):
        canonical_path_ids(invalid)


def test_minimum_hamming_projection_uses_lowest_path_id_for_ties() -> None:
    prediction = np.asarray(
        [[2, 2, 3, 4, 0, 0, 4, 4, 1, 1, 4, 2, 1, 4, 1, 2, 3]],
        dtype=np.int64,
    )
    projected = minimum_hamming_projection(prediction)
    assert projected.shape == prediction.shape
    assert np.array_equal(projected, canonical_paths()[26:27])


def test_path_scores_and_decoder_select_planted_path() -> None:
    grammar = canonical_paths()
    planted = 37
    logits = torch.full((2, 17, 5), -4.0)
    for scale, state in enumerate(grammar[planted]):
        logits[:, scale, int(state)] = 4.0
    scores = path_scores(logits)
    decoded = grammar_decode(logits)
    assert scores.shape == (2, 101)
    assert torch.equal(decoded, torch.as_tensor(grammar[[planted, planted]]))


def test_structured_path_nll_is_differentiable_and_rewards_true_path() -> None:
    grammar = canonical_paths()
    planted = 70
    logits = torch.zeros((1, 17, 5), requires_grad=True)
    baseline = structured_path_nll(logits, torch.tensor([planted]))
    favored = logits.detach().clone()
    for scale, state in enumerate(grammar[planted]):
        favored[0, scale, int(state)] = 3.0
    improved = structured_path_nll(favored, torch.tensor([planted]))
    assert improved < baseline
    baseline.backward()
    assert logits.grad is not None


def test_grammar_decode_removes_invalid_multi_transition_path() -> None:
    logits = torch.full((1, 17, 5), -3.0)
    logits[0, :5, 0] = 3.0
    logits[0, 5:10, 1] = 3.0
    logits[0, 10:, 4] = 3.0
    unconstrained = logits.argmax(dim=2).numpy()
    assert path_validity_rates(unconstrained) == (1.0, 1.0)
    constrained = grammar_decode(logits).numpy()
    assert path_validity_rates(constrained) == (0.0, 0.0)

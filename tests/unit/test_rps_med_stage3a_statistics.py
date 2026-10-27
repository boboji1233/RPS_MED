from __future__ import annotations

import numpy as np

from rps_med.stage3a_statistics import holm_adjust, node_delete_one_comparison


def test_node_delete_one_jackknife_uses_nodes_not_pair_rows() -> None:
    left, right = np.triu_indices(4, k=1)
    truth = np.tile(np.arange(17, dtype=np.int64) % 5, (left.size, 1))
    full = truth.copy()
    comparator = np.full_like(truth, 4)
    result = node_delete_one_comparison(
        [truth, truth, truth],
        [full, full, full],
        [comparator, comparator, comparator],
        left.astype(np.int64),
        right.astype(np.int64),
        metric="macro_f1",
    )
    assert result["node_count"] == 4.0
    assert result["observed_effect"] > 0.0
    assert result["lower_95"] <= result["observed_effect"] <= result["upper_95"]


def test_phase_tv_effect_is_oriented_so_positive_favors_full() -> None:
    left, right = np.triu_indices(4, k=1)
    truth = np.tile(np.arange(17, dtype=np.int64) % 5, (left.size, 1))
    full = truth.copy()
    comparator = np.full_like(truth, 4)
    result = node_delete_one_comparison(
        [truth, truth, truth],
        [full, full, full],
        [comparator, comparator, comparator],
        left.astype(np.int64),
        right.astype(np.int64),
        metric="phase_total_variation",
    )
    assert result["observed_effect"] > 0.0


def test_holm_adjustment_is_monotone_and_never_smaller_than_raw() -> None:
    raw = [0.01, 0.04, 0.03]
    adjusted = holm_adjust(raw)
    assert all(adj >= value for adj, value in zip(adjusted, raw, strict=True))
    ordered = sorted(zip(raw, adjusted, strict=True))
    assert [value for _, value in ordered] == sorted(value for _, value in ordered)

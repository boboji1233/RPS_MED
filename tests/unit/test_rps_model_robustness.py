from __future__ import annotations

import numpy as np

from rps_med.model_robustness import orientation_reversal_consistency


def test_orientation_reversal_swaps_only_directional_relations() -> None:
    forward = np.asarray([[0, 1, 2, 3, 4]], dtype=np.int64)
    reverse = np.asarray([[1, 0, 2, 3, 4]], dtype=np.int64)
    assert orientation_reversal_consistency(forward, reverse) == 1.0
    reverse[0, 3] = 4
    assert orientation_reversal_consistency(forward, reverse) == 0.8

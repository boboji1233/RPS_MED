from __future__ import annotations

import numpy as np

from rps_med.pair_bootstrap_audit import distinct_content_bootstrap_macro_f1_gain


def test_distinct_content_bootstrap_ignores_diagonal_predictions() -> None:
    count = 6
    left, right = np.triu_indices(count, k=0)
    diagonal = left == right
    truth = np.zeros((left.size, 2), dtype=np.int64)
    truth[diagonal] = 2
    baseline = np.ones_like(truth)
    first = truth.copy()
    second = truth.copy()
    second[diagonal] = 4

    first_gain = distinct_content_bootstrap_macro_f1_gain(
        truth,
        first,
        baseline,
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        instance_count=count,
        replicates=20,
        seed=17,
    )
    second_gain = distinct_content_bootstrap_macro_f1_gain(
        truth,
        second,
        baseline,
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        instance_count=count,
        replicates=20,
        seed=17,
    )

    assert np.array_equal(first_gain, second_gain)

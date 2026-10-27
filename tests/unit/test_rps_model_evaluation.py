from __future__ import annotations

import numpy as np

from rps_med.model_evaluation import paired_image_bootstrap_macro_f1_gain


def test_paired_image_bootstrap_detects_perfect_model_gain() -> None:
    instance_count = 6
    left, right = np.triu_indices(instance_count, k=0)
    truth = np.zeros((left.size, 3), dtype=np.int64)
    truth[left == right] = 2
    model = truth.copy()
    baseline = np.full_like(truth, 4)

    gains = paired_image_bootstrap_macro_f1_gain(
        truth,
        model,
        baseline,
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        instance_count=instance_count,
        replicates=20,
        seed=9,
    )

    assert gains.shape == (20,)
    assert bool((gains > 0.0).all())
    assert np.array_equal(
        gains,
        paired_image_bootstrap_macro_f1_gain(
            truth,
            model,
            baseline,
            np.asarray(left, dtype=np.int64),
            np.asarray(right, dtype=np.int64),
            instance_count=instance_count,
            replicates=20,
            seed=9,
        ),
    )

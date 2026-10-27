from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rps_med.model_metrics import PairTargets
from rps_med.stage3b_execution import _train_grammar_head
from rps_med.stage3b_grammar import canonical_paths


def _targets(path_ids: list[int]) -> PairTargets:
    paths = canonical_paths()[path_ids].copy()
    count = len(path_ids)
    return PairTargets(
        left_indices=np.arange(count, dtype=np.int64),
        right_indices=np.arange(1, count + 1, dtype=np.int64),
        relation_codes=paths,
        transition=np.zeros(count, dtype=np.float32),
        critical_log2=np.full(count, np.nan, dtype=np.float32),
    )


def test_synthetic_structured_head_trains_and_decodes_only_valid_paths(
    tmp_path: Path,
) -> None:
    embeddings = np.random.default_rng(7).normal(size=(8, 512)).astype(np.float32)
    train = _targets([0, 1, 2, 5, 21, 37])
    validation = _targets([3, 4, 53, 69, 85])
    config = {
        "training": {
            "dropout": 0.0,
            "batch_size": 3,
            "max_epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "maximum_class_weight": 5.0,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.001,
        },
        "methods": {
            "G3": {
                "structured_weight": 1.0,
                "scale_ce_weight": 0.0,
            }
        },
    }
    artifact = _train_grammar_head(
        "G3",
        embeddings,
        train,
        validation,
        seed=20260826,
        scales=tuple(2.0 ** (-2.0 + index * 0.25) for index in range(17)),
        config=config,
        output_directory=tmp_path,
        device=torch.device("cpu"),
    )
    assert artifact.best_epoch in {1, 2}
    assert artifact.metrics.invalid_spectrum_rate == 0.0
    assert artifact.metrics.multi_transition_rate == 0.0
    assert artifact.resource.peak_cuda_vram_bytes == 0
    assert (tmp_path / "head_G3_seed20260826.pt").is_file()

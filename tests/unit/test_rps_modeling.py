from __future__ import annotations

import numpy as np
import pytest
import torch

from nts_med.relations import RelationConfig
from rps_med.model_metrics import (
    binary_auc,
    build_relation_targets,
    multiscale_macro_f1,
    phase_total_variation,
)
from rps_med.modeling import BurdenEncoder, PairSpectrumHead


def _config() -> RelationConfig:
    return RelationConfig(
        delta={axis: 0.2 for axis in ("ma", "hem", "ex", "cws")},
        epsilon={axis: 0.1 for axis in ("ma", "hem", "ex", "cws")},
        max_width={axis: 0.0 for axis in ("ma", "hem", "ex", "cws")},
    )


def test_model_output_shapes() -> None:
    encoder = BurdenEncoder()
    embedding, burden, presence = encoder(torch.zeros(2, 3, 64, 64))
    assert embedding.shape == (2, 512)
    assert burden.shape == (2, 4)
    assert presence.shape == (2, 4)
    head = PairSpectrumHead(scale_count=3)
    spectrum, transition, critical = head(embedding, embedding.flip(0))
    assert spectrum.shape == (2, 3, 5)
    assert transition.shape == (2,)
    assert critical.shape == (2,)


def test_pair_targets_and_metrics() -> None:
    burdens = np.asarray(
        [[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 1]], dtype=np.float32
    )
    targets = build_relation_targets(
        burdens,
        (0, 1, 2),
        base_config=_config(),
        scales=(0.5, 1.0, 2.0),
    )
    assert targets.pair_count == 3
    assert targets.relation_codes.shape == (3, 3)
    assert multiscale_macro_f1(targets.relation_codes, targets.relation_codes) == 1.0
    assert phase_total_variation(targets.relation_codes, targets.relation_codes) == 0.0


def test_binary_auc() -> None:
    truth = np.asarray([0, 0, 1, 1], dtype=np.float32)
    score = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    assert binary_auc(truth, score) == pytest.approx(1.0)

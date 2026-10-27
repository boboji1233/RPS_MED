from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision.models import resnet18  # type: ignore[import-untyped]

from rps_med.model_training import run_tjdr_model_development


def test_synthetic_model_development_stays_official_test_free(tmp_path: Path) -> None:
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    rng = np.random.default_rng(4)
    image_count = 12
    cache = tmp_path / "cache.npz"
    burdens = np.asarray(
        [
            [index * 0.2, (index % 3) * 0.4, (index % 4) * 0.3, (index % 5) * 0.2]
            for index in range(image_count)
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        cache,
        images=rng.integers(0, 256, (image_count, 64, 64, 3), dtype=np.uint8),
        burdens=burdens,
        presence=np.asarray(burdens > 0.0, dtype=np.float32),
        partition=np.asarray([0] * 8 + [1] * 4, dtype=np.uint8),
        annotation_group_size=np.ones(image_count, dtype=np.uint8),
        source_shape_code=np.zeros(image_count, dtype=np.uint8),
        axis_names=np.asarray(("ma", "hem", "ex", "cws")),
        image_size=np.asarray([64], dtype=np.int32),
    )
    pretrained = tmp_path / "resnet18.pth"
    torch.save(resnet18(weights=None).state_dict(), pretrained)
    scales = [2 ** (-2 + index * 0.25) for index in range(17)]
    config = {
        "relation": {
            "scales": scales,
            "thresholds": {
                key: {axis: value for axis in ("ma", "hem", "ex", "cws")}
                for key, value in (("delta", 0.2), ("epsilon", 0.1), ("max_width", 0.0))
            }
            | {"numeric_tolerance": 1.0e-12},
        },
        "seeds": [17],
        "burden_model": {
            "batch_size": 4,
            "max_epochs": 1,
            "encoder_lr": 0.0001,
            "head_lr": 0.0005,
            "weight_decay": 0.0001,
            "huber_beta": 0.5,
            "presence_loss_weight": 0.25,
            "early_stopping_min_delta": 0.0,
            "early_stopping_patience": 1,
            "amp": False,
        },
        "pair_model": {
            "batch_size": 32,
            "max_epochs": 1,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "maximum_class_weight": 5.0,
            "transition_loss_weight": 0.25,
            "critical_loss_weight": 0.25,
            "critical_huber_beta": 0.5,
            "early_stopping_min_delta": 0.0,
            "early_stopping_patience": 1,
            "amp": False,
        },
        "development_gate": {"minimum_macro_f1_margin_over_majority": 0.05},
        "runtime": {"device": "cpu"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_tjdr_model_development(
        cache,
        pretrained,
        config_path,
        tmp_path / "weights",
    )

    assert result["status"] == "RPS_MED_TJDR_MODEL_DEVELOPMENT_COMPLETE"
    assert result["official_test_accessed"] is False
    assert result["development"]["train_pair_count"] == 28
    assert result["development"]["validation_pair_count"] == 6

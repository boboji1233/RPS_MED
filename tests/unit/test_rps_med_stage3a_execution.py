from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from rps_med.model_metrics import PairTargets
from rps_med.stage3a_execution import _train_pair_run, run_stage3a_development

ROOT = Path(__file__).resolve().parents[2]


def _targets(node_offset: int, pair_count: int) -> PairTargets:
    left = np.arange(pair_count, dtype=np.int64) % 4 + node_offset
    right = (left + 1 - node_offset) % 4 + node_offset
    relation = np.empty((pair_count, 17), dtype=np.int64)
    for row in range(pair_count):
        relation[row] = (row + np.arange(17)) % 5
    transition = np.asarray([(row % 2) for row in range(pair_count)], dtype=np.float32)
    critical = np.full(pair_count, np.nan, dtype=np.float32)
    critical[transition > 0.5] = 0.5
    return PairTargets(
        left_indices=left,
        right_indices=right,
        relation_codes=relation,
        transition=transition,
        critical_log2=critical,
    )


def test_stage3a_pair_training_writes_one_append_only_checkpoint(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260826)
    embeddings = rng.normal(size=(8, 512)).astype(np.float32)
    common = {
        "pair_training": {
            "dropout": 0.0,
            "batch_size": 4,
            "max_epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "maximum_class_weight": 5.0,
            "transition_loss_weight": 0.25,
            "critical_loss_weight": 0.25,
            "critical_huber_beta": 0.5,
            "soft_admissibility_weight": 0.05,
            "early_stopping_min_delta": 0.001,
            "early_stopping_patience": 2,
            "amp": False,
        }
    }
    artifact = _train_pair_run(
        "SYNTHETIC",
        {
            "head": "joint_spectrum",
            "losses": ["spectrum", "transition", "critical"],
        },
        embeddings,
        _targets(0, 8),
        _targets(4, 4),
        output_directory=tmp_path,
        seed=20260826,
        device=torch.device("cpu"),
        common=common,
        scales=tuple(2.0 ** (-2.0 + index * 0.25) for index in range(17)),
    )
    assert artifact.best_epoch in {1, 2}
    assert artifact.prediction.shape == (4, 17)
    assert artifact.reverse_prediction.shape == (4, 17)
    assert artifact.parameter_count > 0
    assert len(tuple(tmp_path.glob("*.pt"))) == 1


def test_stage3a_runner_refuses_existing_or_out_of_scope_output(tmp_path: Path) -> None:
    common = ROOT / "configs" / "stage3a" / "rps_med_stage3a_common_v1.yaml"
    runs = ROOT / "configs" / "stage3a" / "rps_med_stage3a_runs_v1.yaml"
    with pytest.raises(FileExistsError, match="output root"):
        run_stage3a_development(
            ROOT,
            tmp_path / "cache.npz",
            tmp_path / "weights.pth",
            common,
            runs,
            tmp_path,
            include_p1=False,
        )
    outside = tmp_path / "new-output"
    with pytest.raises(ValueError, match="reports/stage3a"):
        run_stage3a_development(
            ROOT,
            tmp_path / "cache.npz",
            tmp_path / "weights.pth",
            common,
            runs,
            outside,
            include_p1=False,
        )


def test_stage3a_source_imports_no_formal_or_ddr_executor() -> None:
    source = (ROOT / "src" / "rps_med" / "stage3a_execution.py").read_text(
        encoding="utf-8"
    )
    import_lines = [line for line in source.splitlines() if line.startswith(("from ", "import "))]
    assert not any("model_evaluation" in line for line in import_lines)
    assert not any("ddr_model_transfer" in line for line in import_lines)

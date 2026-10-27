from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rps_med.stage3b_reporting import analyze_stage3b


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_stage3b_analysis_uses_nodes_and_applies_frozen_classification(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "EXECUTION_COMPLETE.json",
        {"status": "STAGE3B_DEVELOPMENT_EXECUTION_COMPLETE"},
    )
    left, right = np.triu_indices(4, k=1)
    truth = np.tile(np.arange(17, dtype=np.int64) % 5, (left.size, 1))
    predictions = {
        "G0": np.full_like(truth, 4),
        "G1": truth.copy(),
        "G3": truth.copy(),
        "G4": truth.copy(),
    }
    macro = {"G0": 0.45, "G1": 0.449, "G3": 0.46, "G4": 0.43}
    states = (
        "LESS",
        "GREATER",
        "EQUIVALENT",
        "STRUCTURAL_INCOMPARABLE",
        "UNRESOLVED",
    )
    for method, prediction in predictions.items():
        for seed in (20260826, 20260827, 20260828):
            directory = tmp_path / "runs" / method / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                directory / "validation_predictions.npz",
                truth=truth,
                prediction=prediction,
                reverse_prediction=prediction,
                left_node=left.astype(np.int64),
                right_node=right.astype(np.int64),
            )
            _write_json(
                directory / "result.json",
                {
                    "seed": seed,
                    "metrics": {
                        "spectrum_macro_f1": macro[method],
                        "relation_accuracy": macro[method],
                        "phase_total_variation": 1.0 - macro[method],
                        "pair_swap_consistency": 1.0,
                        "invalid_spectrum_rate": 0.01 if method == "G0" else 0.0,
                        "multi_transition_rate": 0.01 if method == "G0" else 0.0,
                        "per_state_f1": {state: macro[method] for state in states},
                    },
                    "resource": {
                        "runtime_seconds": 1.0,
                        "peak_cuda_vram_bytes": 10.0,
                        "trainable_parameter_count": 0 if method in {"G0", "G1"} else 5,
                        "best_epoch": None if method in {"G0", "G1"} else 2,
                    },
                },
            )
    result = analyze_stage3b(tmp_path)
    assert result["statistical_unit"] == "VALIDATION_UNIQUE_CONTENT_NODE"
    assert len(result["node_level_comparisons"]) == 21
    assert result["classifications"]["G1"]["decision"] == "FIDELITY_PRESERVING"
    assert result["classifications"]["G3"]["decision"] == "FIDELITY_PRESERVING"
    assert result["classifications"]["G4"]["decision"] == "VALIDITY_ONLY"

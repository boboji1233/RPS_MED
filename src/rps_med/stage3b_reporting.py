"""Aggregate reporting for the development-only RPS-Med Stage 3B grammar study."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rps_med.stage3a_statistics import holm_adjust, node_delete_one_comparison

_SEEDS = (20260826, 20260827, 20260828)
_METHODS = ("G0", "G1", "G3", "G4")
_STATES = (
    "LESS",
    "GREATER",
    "EQUIVALENT",
    "STRUCTURAL_INCOMPARABLE",
    "UNRESOLVED",
)


def _json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Stage 3B JSON must be an object")
    return payload


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Stage 3B summary received no values")
    return {
        "mean": mean(values),
        "sample_sd": 0.0 if len(values) == 1 else stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _prediction(root: Path, method: str, seed: int) -> dict[str, NDArray[Any]]:
    path = root / "runs" / method / f"seed_{seed}" / "validation_predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _seed_summaries(root: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method in _METHODS:
        rows = [_json(root / "runs" / method / f"seed_{seed}" / "result.json") for seed in _SEEDS]
        if [int(row["seed"]) for row in rows] != list(_SEEDS):
            raise ValueError("Stage 3B method is missing a frozen seed")
        metric_names = (
            "spectrum_macro_f1",
            "relation_accuracy",
            "phase_total_variation",
            "pair_swap_consistency",
            "invalid_spectrum_rate",
            "multi_transition_rate",
        )
        metrics: dict[str, Any] = {
            name: _summary([float(row["metrics"][name]) for row in rows])
            for name in metric_names
        }
        metrics["per_state_f1"] = {
            state: _summary(
                [float(row["metrics"]["per_state_f1"][state]) for row in rows]
            )
            for state in _STATES
        }
        output[method] = {
            "seeds": list(_SEEDS),
            "metrics": metrics,
            "resource": {
                "runtime_seconds": _summary(
                    [float(row["resource"]["runtime_seconds"]) for row in rows]
                ),
                "peak_cuda_vram_bytes": _summary(
                    [float(row["resource"]["peak_cuda_vram_bytes"]) for row in rows]
                ),
                "trainable_parameter_count": _summary(
                    [float(row["resource"]["trainable_parameter_count"]) for row in rows]
                ),
                "best_epoch": (
                    None
                    if method in {"G0", "G1"}
                    else _summary([float(row["resource"]["best_epoch"]) for row in rows])
                ),
            },
        }
    return output


def _comparisons(root: Path) -> list[dict[str, Any]]:
    reference = [_prediction(root, "G0", seed) for seed in _SEEDS]
    truth = [np.asarray(row["truth"], dtype=np.int64) for row in reference]
    left = np.asarray(reference[0]["left_node"], dtype=np.int64)
    right = np.asarray(reference[0]["right_node"], dtype=np.int64)
    metrics = ("macro_f1", "phase_total_variation", *[f"per_state_f1:{s}" for s in _STATES])
    rows: list[dict[str, Any]] = []
    adjustment_groups: dict[str, list[int]] = {metric: [] for metric in metrics}
    for method in _METHODS[1:]:
        payloads = [_prediction(root, method, seed) for seed in _SEEDS]
        predictions = [np.asarray(row["prediction"], dtype=np.int64) for row in payloads]
        for metric in metrics:
            result = node_delete_one_comparison(
                truth,
                predictions,
                [np.asarray(row["prediction"], dtype=np.int64) for row in reference],
                left,
                right,
                metric=metric,
            )
            rows.append(
                {
                    "method": method,
                    "reference": "G0",
                    "metric": metric,
                    **result,
                    "evidence_level": "DEVELOPMENT_ONLY_SUPPORTIVE_EVIDENCE",
                }
            )
            adjustment_groups[metric].append(len(rows) - 1)
    for indices in adjustment_groups.values():
        adjusted = holm_adjust([float(rows[index]["raw_two_sided_p"]) for index in indices])
        for index, value in zip(indices, adjusted, strict=True):
            rows[index]["holm_adjusted_p"] = value
    return rows


def analyze_stage3b(output_root: str | Path) -> dict[str, Any]:
    """Return preregistered summaries, node comparisons and classifications."""

    root = Path(output_root)
    complete = _json(root / "EXECUTION_COMPLETE.json")
    if complete["status"] != "STAGE3B_DEVELOPMENT_EXECUTION_COMPLETE":
        raise ValueError("Stage 3B execution is not complete")
    summaries = _seed_summaries(root)
    reference_mean = float(summaries["G0"]["metrics"]["spectrum_macro_f1"]["mean"])
    classifications: dict[str, Any] = {}
    floor = -0.010
    for method in _METHODS[1:]:
        metrics = summaries[method]["metrics"]
        delta = float(metrics["spectrum_macro_f1"]["mean"]) - reference_mean
        valid = (
            float(metrics["invalid_spectrum_rate"]["maximum"]) == 0.0
            and float(metrics["multi_transition_rate"]["maximum"]) == 0.0
        )
        if not valid:
            decision = "INVALID_IMPLEMENTATION"
        elif delta >= floor:
            decision = "FIDELITY_PRESERVING"
        else:
            decision = "VALIDITY_ONLY"
        classifications[method] = {
            "decision": decision,
            "mean_macro_f1_delta_versus_G0": delta,
            "fidelity_floor": floor,
            "every_seed_valid": valid,
        }
    return {
        "status": "STAGE3B_DEVELOPMENT_ANALYSIS_COMPLETE",
        "evidence_level": "DEVELOPMENT_ONLY_SUPPORTIVE_EVIDENCE",
        "seed_summaries": summaries,
        "node_level_comparisons": _comparisons(root),
        "classifications": classifications,
        "statistical_unit": "VALIDATION_UNIQUE_CONTENT_NODE",
        "pair_row_bootstrap": False,
        "official_test_accessed": False,
        "ddr_accessed": False,
        "formal_results_recomputed": False,
    }


__all__ = ["analyze_stage3b"]

"""Node-level supportive statistics for RPS-Med Stage 3A development runs."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rps_med.model_metrics import multiscale_macro_f1, phase_total_variation
from rps_med.stage3a_models import pooled_per_state_f1

_STATES = (
    "LESS",
    "GREATER",
    "EQUIVALENT",
    "STRUCTURAL_INCOMPARABLE",
    "UNRESOLVED",
)


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Stage 3A result JSON must be an object")
    return payload


def _sample_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Stage 3A seed summary is empty")
    return {
        "mean": mean(values),
        "sample_sd": 0.0 if len(values) == 1 else stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize_seed_results(output_root: str | Path) -> dict[str, Any]:
    """Aggregate append-only result JSON files without opening model weights."""

    root = Path(output_root)
    rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for path in sorted((root / "runs").glob("*/seed_*/result.json")):
        payload = _load_json(path)
        rows[str(payload["run_id"])].append(payload)
    output: dict[str, Any] = {}
    for run_id, run_rows in sorted(rows.items()):
        run_rows = sorted(run_rows, key=lambda row: int(row["seed"]))
        seeds = [int(row["seed"]) for row in run_rows]
        if seeds != [20260826, 20260827, 20260828]:
            raise ValueError("Stage 3A run is missing a frozen seed")
        metric_names = (
            "spectrum_macro_f1",
            "relation_accuracy",
            "phase_total_variation",
            "pair_swap_consistency",
            "invalid_spectrum_rate",
            "multi_transition_rate",
        )
        metrics: dict[str, Any] = {}
        for name in metric_names:
            values = [row["metrics"][name] for row in run_rows]
            if any(value is None for value in values):
                metrics[name] = None
            else:
                metrics[name] = _sample_summary([float(value) for value in values])
        for name in ("transition_auc", "critical_log2_mae"):
            values = [row["metrics"][name] for row in run_rows]
            finite = [float(value) for value in values if value is not None]
            metrics[name] = None if len(finite) != len(values) else _sample_summary(finite)
        per_state: dict[str, Any] = {}
        state_names = tuple(run_rows[0]["metrics"]["per_state_f1"])
        for state in state_names:
            values = [row["metrics"]["per_state_f1"][state] for row in run_rows]
            finite = [float(value) for value in values if value is not None]
            per_state[state] = None if len(finite) != len(values) else _sample_summary(finite)
        metrics["per_state_f1"] = per_state
        resource_names = (
            "total_pipeline_runtime_seconds",
            "peak_cuda_vram_bytes",
            "total_pipeline_parameter_count",
            "best_encoder_epoch",
            "best_head_epoch",
        )
        resources: dict[str, Any] = {}
        for name in resource_names:
            values = [row["resource"][name] for row in run_rows]
            finite = [float(value) for value in values if value is not None]
            resources[name] = None if len(finite) != len(values) else _sample_summary(finite)
        output[run_id] = {
            "aliases": run_rows[0]["aliases"],
            "tier": run_rows[0]["tier"],
            "family": run_rows[0]["family"],
            "seeds": seeds,
            "metrics": metrics,
            "resource": resources,
            "run_contract_sha256": [
                row["provenance"]["run_contract_sha256"] for row in run_rows
            ],
        }
    return output


def _prediction_payload(
    output_root: Path, run_id: str, seed: int
) -> dict[str, NDArray[Any]]:
    path = output_root / "runs" / run_id / f"seed_{seed}" / "validation_predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _metric_value(
    metric: str,
    truth: NDArray[np.int64],
    prediction: NDArray[np.int64],
) -> float:
    if metric == "macro_f1":
        return multiscale_macro_f1(truth, prediction)
    if metric == "phase_total_variation":
        return phase_total_variation(truth, prediction)
    if metric.startswith("per_state_f1:"):
        state = metric.split(":", 1)[1]
        value = pooled_per_state_f1(truth, prediction)[state]
        if value is None:
            raise ValueError("Stage 3A comparison state is unsupported")
        return value
    raise ValueError("unknown Stage 3A comparison metric")


def _effect(
    metric: str,
    truth: NDArray[np.int64],
    full: NDArray[np.int64],
    comparator: NDArray[np.int64],
) -> float:
    full_value = _metric_value(metric, truth, full)
    comparator_value = _metric_value(metric, truth, comparator)
    if metric == "phase_total_variation":
        return comparator_value - full_value
    return full_value - comparator_value


def node_delete_one_comparison(
    truth_by_seed: Sequence[NDArray[np.int64]],
    full_by_seed: Sequence[NDArray[np.int64]],
    comparator_by_seed: Sequence[NDArray[np.int64]],
    left: NDArray[np.int64],
    right: NDArray[np.int64],
    *,
    metric: str,
) -> dict[str, float]:
    """Return a paired unique-node delete-one jackknife comparison."""

    if not (
        len(truth_by_seed) == len(full_by_seed) == len(comparator_by_seed) == 3
        and left.shape == right.shape
    ):
        raise ValueError("Stage 3A jackknife inputs do not align")
    nodes = np.unique(np.concatenate((left, right)))
    observed_seed = [
        _effect(metric, truth, full, comparator)
        for truth, full, comparator in zip(
            truth_by_seed, full_by_seed, comparator_by_seed, strict=True
        )
    ]
    observed = float(np.mean(observed_seed))
    delete_values: list[float] = []
    for node in nodes:
        keep = np.logical_and(left != node, right != node)
        if not bool(keep.any()):
            raise ValueError("node deletion removed every validation edge")
        seed_values = [
            _effect(metric, truth[keep], full[keep], comparator[keep])
            for truth, full, comparator in zip(
                truth_by_seed, full_by_seed, comparator_by_seed, strict=True
            )
        ]
        delete_values.append(float(np.mean(seed_values)))
    delete = np.asarray(delete_values, dtype=np.float64)
    center = float(delete.mean())
    standard_error = math.sqrt(
        (nodes.size - 1) / nodes.size * float(np.square(delete - center).sum())
    )
    if standard_error == 0.0:
        p_value = 0.0 if observed != 0.0 else 1.0
    else:
        p_value = math.erfc(abs(observed / standard_error) / math.sqrt(2.0))
    return {
        "observed_effect": observed,
        "jackknife_standard_error": standard_error,
        "lower_95": observed - 1.96 * standard_error,
        "upper_95": observed + 1.96 * standard_error,
        "raw_two_sided_p": p_value,
        "node_count": float(nodes.size),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in original order."""

    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _comparison_family(
    output_root: Path,
    *,
    family: str,
    full_run: str,
    comparators: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    seeds = (20260826, 20260827, 20260828)
    full_payloads = [_prediction_payload(output_root, full_run, seed) for seed in seeds]
    left = np.asarray(full_payloads[0]["left_node"], dtype=np.int64)
    right = np.asarray(full_payloads[0]["right_node"], dtype=np.int64)
    truth = [np.asarray(payload["truth"], dtype=np.int64) for payload in full_payloads]
    full = [np.asarray(payload["prediction"], dtype=np.int64) for payload in full_payloads]
    rows: list[dict[str, Any]] = []
    by_adjustment: dict[str, list[int]] = defaultdict(list)
    for comparator in comparators:
        comparator_payloads = [
            _prediction_payload(output_root, comparator, seed) for seed in seeds
        ]
        comparator_predictions = [
            np.asarray(payload["prediction"], dtype=np.int64)
            for payload in comparator_payloads
        ]
        for metric in metrics:
            comparison = node_delete_one_comparison(
                truth,
                full,
                comparator_predictions,
                left,
                right,
                metric=metric,
            )
            adjustment_family = f"{family}:{metric}"
            rows.append(
                {
                    "family": family,
                    "adjustment_family": adjustment_family,
                    "full_run": full_run,
                    "comparator_run": comparator,
                    "metric": metric,
                    **comparison,
                    "evidence_level": "DEVELOPMENT_ONLY_SUPPORTIVE_EVIDENCE",
                }
            )
            by_adjustment[adjustment_family].append(len(rows) - 1)
    for indices in by_adjustment.values():
        adjusted = holm_adjust([float(rows[index]["raw_two_sided_p"]) for index in indices])
        for index, value in zip(indices, adjusted, strict=True):
            rows[index]["holm_adjusted_p"] = value
    return rows


def analyze_stage3a(output_root: str | Path) -> dict[str, Any]:
    """Create the preregistered seed summaries and node-level comparisons."""

    root = Path(output_root)
    complete = _load_json(root / "EXECUTION_COMPLETE.json")
    if complete["status"] != "STAGE3A_DEVELOPMENT_EXECUTION_COMPLETE":
        raise ValueError("Stage 3A execution is not complete")
    summaries = summarize_seed_results(root)
    metrics = ("macro_f1", "phase_total_variation", *[f"per_state_f1:{s}" for s in _STATES])
    p0 = _comparison_family(
        root,
        family="P0_FULL_VERSUS_BASELINES",
        full_run="B6_A0_A12full",
        comparators=("B0", "B1", "B2_A1", "B3_A3", "B4", "B5"),
        metrics=metrics,
    )
    ablation = _comparison_family(
        root,
        family="P0_FULL_VERSUS_ABLATIONS",
        full_run="B6_A0_A12full",
        comparators=(
            "B2_A1",
            "A2",
            "B3_A3",
            "A4",
            "A5",
            "A6",
            "A7",
            "A8",
            "A9",
            "A10",
            "A12_signed",
            "A12_absolute",
        ),
        metrics=metrics,
    )
    return {
        "status": "STAGE3A_NODE_LEVEL_ANALYSIS_COMPLETE",
        "evidence_level": "DEVELOPMENT_ONLY_SUPPORTIVE_EVIDENCE",
        "seed_summaries": summaries,
        "node_level_comparisons": [*p0, *ablation],
        "statistics_unit": "VALIDATION_UNIQUE_CONTENT_NODE",
        "pair_row_bootstrap": False,
        "official_test_accessed": False,
        "ddr_accessed": False,
        "formal_results_recomputed": False,
    }


__all__ = [
    "analyze_stage3a",
    "holm_adjust",
    "node_delete_one_comparison",
    "summarize_seed_results",
]

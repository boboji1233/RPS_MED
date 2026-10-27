"""Development-only estimand audit for complete-graph pair resampling."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from numpy.typing import NDArray

from rps_med.model_data import load_tjdr_model_cache
from rps_med.model_evaluation import (
    _image_embeddings,
    _pair_ensemble,
    _truth_with_diagonal,
    _weighted_macro_f1,
    paired_image_bootstrap_macro_f1_gain,
)
from rps_med.model_metrics import (
    build_relation_targets,
    majority_spectrum_baseline,
    multiscale_macro_f1,
)
from rps_med.model_training import _relation_config, _scales, _seed_everything


def distinct_content_bootstrap_macro_f1_gain(
    truth: NDArray[np.int64],
    model_prediction: NDArray[np.int64],
    baseline_prediction: NDArray[np.int64],
    left: NDArray[np.int64],
    right: NDArray[np.int64],
    *,
    instance_count: int,
    replicates: int,
    seed: int,
) -> NDArray[np.float64]:
    """Bootstrap image weights while retaining only distinct source-content pairs."""

    if not (
        truth.shape == model_prediction.shape == baseline_prediction.shape
        and truth.shape[0] == left.size == right.size
    ):
        raise ValueError("distinct-content bootstrap arrays do not align")
    rng = np.random.default_rng(seed)
    output = np.empty(replicates, dtype=np.float64)
    diagonal = left == right
    for replicate in range(replicates):
        counts = rng.multinomial(
            instance_count, np.full(instance_count, 1.0 / instance_count)
        )
        weights = np.asarray(counts[left] * counts[right], dtype=np.float64)
        weights[diagonal] = 0.0
        if float(weights.sum()) <= 0.0:
            raise RuntimeError("distinct-content bootstrap has no weighted pair")
        output[replicate] = _weighted_macro_f1(
            truth, model_prediction, weights
        ) - _weighted_macro_f1(truth, baseline_prediction, weights)
    return output


def delete_one_image_jackknife_gain(
    truth: NDArray[np.int64],
    model_prediction: NDArray[np.int64],
    baseline_prediction: NDArray[np.int64],
    left: NDArray[np.int64],
    right: NDArray[np.int64],
    *,
    instance_count: int,
) -> dict[str, float]:
    """Return a node-level delete-one jackknife summary for the pair statistic."""

    estimates = np.empty(instance_count, dtype=np.float64)
    for index in range(instance_count):
        keep = np.logical_and(left != index, right != index)
        estimates[index] = multiscale_macro_f1(
            truth[keep], model_prediction[keep]
        ) - multiscale_macro_f1(truth[keep], baseline_prediction[keep])
    mean = float(np.mean(estimates))
    variance = (instance_count - 1) / instance_count * float(
        np.square(estimates - mean).sum()
    )
    standard_error = math.sqrt(max(variance, 0.0))
    point = multiscale_macro_f1(truth, model_prediction) - multiscale_macro_f1(
        truth, baseline_prediction
    )
    return {
        "point_estimate": point,
        "leave_one_mean": mean,
        "standard_error": standard_error,
        "normal_lower_95": point - 1.959963984540054 * standard_error,
        "normal_upper_95": point + 1.959963984540054 * standard_error,
    }


def _distribution_summary(values: NDArray[np.float64], point: float) -> dict[str, Any]:
    lower, upper = np.quantile(values, (0.025, 0.975))
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "point_estimate_inside_percentile_interval": bool(lower <= point <= upper),
        "mean_minus_point": float(np.mean(values) - point),
    }


def audit_development_pair_bootstrap(
    cache_path: str | Path,
    config_path: str | Path,
    weights_directory: str | Path,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Compare pair-resampling estimands using development validation only."""

    cache = load_tjdr_model_cache(cache_path)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    images = np.asarray(cache["images"], dtype=np.uint8)
    burdens = np.asarray(cache["burdens"], dtype=np.float32)
    partition = np.asarray(cache["partition"], dtype=np.uint8)
    train_indices = np.flatnonzero(partition == 0).astype(np.int64)
    validation_indices = np.flatnonzero(partition == 1).astype(np.int64)
    relation_config = _relation_config(config)
    scales = _scales(config)
    train_truth = build_relation_targets(
        burdens,
        train_indices.tolist(),
        base_config=relation_config,
        scales=scales,
    )
    validation_truth = build_relation_targets(
        burdens,
        validation_indices.tolist(),
        base_config=relation_config,
        scales=scales,
    )
    seeds = tuple(int(value) for value in config["seeds"])
    weights_root = Path(weights_directory)
    burden_paths = tuple(weights_root / f"burden_encoder_seed{value}.pt" for value in seeds)
    head_paths = tuple(weights_root / f"rps_multitask_pair_seed{value}.pt" for value in seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("pair bootstrap audit requires the frozen CUDA runtime")
    _seed_everything(seed)
    embeddings = tuple(
        _image_embeddings(
            images[validation_indices],
            path,
            batch_size=int(config["burden_model"]["batch_size"]),
            device=device,
        )
        for path in burden_paths
    )
    left, right, truth, diagonal = _truth_with_diagonal(
        validation_truth, validation_indices.size, len(scales)
    )
    prediction, _, _, _ = _pair_ensemble(
        embeddings,
        head_paths,
        left,
        right,
        scale_count=len(scales),
        batch_size=int(config["pair_model"]["batch_size"]),
        device=device,
    )
    baseline_off = majority_spectrum_baseline(train_truth, validation_truth.pair_count)
    baseline = np.tile(baseline_off[0], (truth.shape[0], 1))
    point = multiscale_macro_f1(
        validation_truth.relation_codes, prediction[~diagonal]
    ) - multiscale_macro_f1(validation_truth.relation_codes, baseline_off)
    frozen = paired_image_bootstrap_macro_f1_gain(
        truth,
        prediction,
        baseline,
        left,
        right,
        instance_count=validation_indices.size,
        replicates=replicates,
        seed=seed,
    )
    distinct = distinct_content_bootstrap_macro_f1_gain(
        truth,
        prediction,
        baseline,
        left,
        right,
        instance_count=validation_indices.size,
        replicates=replicates,
        seed=seed,
    )
    off_left = left[~diagonal]
    off_right = right[~diagonal]
    jackknife = delete_one_image_jackknife_gain(
        validation_truth.relation_codes,
        prediction[~diagonal],
        baseline_off,
        off_left,
        off_right,
        instance_count=validation_indices.size,
    )
    return {
        "status": "RPS_MED_PAIR_BOOTSTRAP_DEVELOPMENT_AUDIT_COMPLETE",
        "scope": "TJDR_DEVELOPMENT_VALIDATION_ONLY",
        "independent_unit": "UNIQUE_IMAGE_CONTENT",
        "validation_images": int(validation_indices.size),
        "ordinary_unordered_pairs": validation_truth.pair_count,
        "observed_macro_f1_gain": point,
        "frozen_bootstrap_with_resampled_self_content_pairs": _distribution_summary(
            frozen, point
        ),
        "distinct_content_weighted_bootstrap": _distribution_summary(distinct, point),
        "delete_one_unique_image_jackknife": jackknife,
        "replicates": replicates,
        "seed": seed,
        "official_test_accessed": False,
        "ddr_scientific_data_accessed": False,
        "model_retraining": False,
        "row_predictions_persisted": False,
        "source_identifiers_recorded": False,
    }


__all__ = [
    "audit_development_pair_bootstrap",
    "delete_one_image_jackknife_gain",
    "distinct_content_bootstrap_macro_f1_gain",
]

"""One-run official-test evaluation for the frozen TJDR RPS-Med model."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from numpy.typing import NDArray

from nts_med.relations import Relation
from rps_med.model_data import load_tjdr_model_cache, load_tjdr_test_model_arrays
from rps_med.model_metrics import (
    PairTargets,
    binary_auc,
    build_relation_targets,
    critical_log2_mae,
    majority_spectrum_baseline,
    multiscale_macro_f1,
    phase_total_variation,
    relation_accuracy,
)
from rps_med.model_training import _relation_config, _scales, _seed_everything
from rps_med.modeling import BurdenEncoder, PairSpectrumHead

_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)


def _image_embeddings(
    images: NDArray[np.uint8],
    checkpoint_path: Path,
    *,
    batch_size: int,
    device: torch.device,
) -> NDArray[np.float32]:
    model = BurdenEncoder().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    outputs: list[NDArray[np.float32]] = []
    mean = _MEAN.to(device)
    std = _STD.to(device)
    with torch.inference_mode():
        for start in range(0, images.shape[0], batch_size):
            batch = torch.from_numpy(images[start : start + batch_size].copy())
            batch = batch.permute(0, 3, 1, 2).float().to(device) / 255.0
            embedding, _, _ = model((batch - mean) / std)
            outputs.append(embedding.float().cpu().numpy())
    return np.concatenate(outputs)


def _upper_triangle_indices(count: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    left, right = np.triu_indices(count, k=0)
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def _pair_ensemble(
    embeddings_by_seed: Sequence[NDArray[np.float32]],
    head_paths: Sequence[Path],
    left: NDArray[np.int64],
    right: NDArray[np.int64],
    *,
    scale_count: int,
    batch_size: int,
    device: torch.device,
) -> tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.float32], list[NDArray[np.int64]]]:
    probability_sum = np.zeros((left.size, scale_count, len(Relation)), dtype=np.float64)
    transition_sum = np.zeros(left.size, dtype=np.float64)
    critical_sum = np.zeros(left.size, dtype=np.float64)
    per_seed_predictions: list[NDArray[np.int64]] = []
    for embeddings_array, head_path in zip(embeddings_by_seed, head_paths, strict=True):
        checkpoint = torch.load(head_path, map_location=device, weights_only=False)
        head = PairSpectrumHead(scale_count=scale_count, relation_count=len(Relation)).to(device)
        head.load_state_dict(checkpoint["model"])
        head.eval()
        embeddings = torch.from_numpy(embeddings_array).to(device)
        seed_probabilities: list[NDArray[np.float32]] = []
        seed_transitions: list[NDArray[np.float32]] = []
        seed_critical: list[NDArray[np.float32]] = []
        with torch.inference_mode():
            for start in range(0, left.size, batch_size):
                selector = slice(start, start + batch_size)
                logits, transition, critical_standardized = head(
                    embeddings[torch.from_numpy(left[selector]).to(device)],
                    embeddings[torch.from_numpy(right[selector]).to(device)],
                )
                seed_probabilities.append(torch.softmax(logits, dim=2).float().cpu().numpy())
                seed_transitions.append(torch.sigmoid(transition).float().cpu().numpy())
                seed_critical.append(critical_standardized.float().cpu().numpy())
        probabilities = np.concatenate(seed_probabilities)
        transition_scores = np.concatenate(seed_transitions)
        critical_standardized = np.concatenate(seed_critical)
        critical = (
            critical_standardized * float(checkpoint["critical_std"])
            + float(checkpoint["critical_mean"])
        )
        probability_sum += probabilities
        transition_sum += transition_scores
        critical_sum += critical
        per_seed_predictions.append(np.asarray(probabilities.argmax(axis=2), dtype=np.int64))
    seed_count = len(head_paths)
    ensemble_probability = probability_sum / seed_count
    return (
        np.asarray(ensemble_probability.argmax(axis=2), dtype=np.int64),
        np.asarray(transition_sum / seed_count, dtype=np.float32),
        np.asarray(critical_sum / seed_count, dtype=np.float32),
        per_seed_predictions,
    )


def _truth_with_diagonal(
    off_diagonal: PairTargets, instance_count: int, scale_count: int
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]:
    left, right = _upper_triangle_indices(instance_count)
    diagonal = left == right
    truth = np.full((left.size, scale_count), len(Relation) - 1, dtype=np.int64)
    truth[diagonal] = tuple(Relation).index(Relation.EQUIVALENT)
    truth[~diagonal] = off_diagonal.relation_codes
    return left, right, truth, diagonal


def _weighted_macro_f1(
    truth: NDArray[np.int64], prediction: NDArray[np.int64], weights: NDArray[np.float64]
) -> float:
    scale_scores: list[float] = []
    for scale_index in range(truth.shape[1]):
        class_scores: list[float] = []
        for code in range(len(Relation)):
            actual = truth[:, scale_index] == code
            actual_weight = float(weights[actual].sum())
            if actual_weight <= 0.0:
                continue
            predicted = prediction[:, scale_index] == code
            true_positive = float(weights[np.logical_and(actual, predicted)].sum())
            false_positive = float(weights[np.logical_and(~actual, predicted)].sum())
            false_negative = float(weights[np.logical_and(actual, ~predicted)].sum())
            denominator = 2.0 * true_positive + false_positive + false_negative
            class_scores.append(0.0 if denominator <= 0.0 else 2.0 * true_positive / denominator)
        scale_scores.append(float(np.mean(class_scores)))
    return float(np.mean(scale_scores))


def paired_image_bootstrap_macro_f1_gain(
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
    """Bootstrap images and rebuild all unordered pairs through edge weights."""

    if not (
        truth.shape == model_prediction.shape == baseline_prediction.shape
        and truth.shape[0] == left.size == right.size
    ):
        raise ValueError("bootstrap pair arrays do not align")
    rng = np.random.default_rng(seed)
    output = np.empty(replicates, dtype=np.float64)
    diagonal = left == right
    for replicate in range(replicates):
        counts = rng.multinomial(instance_count, np.full(instance_count, 1.0 / instance_count))
        weights = np.asarray(counts[left] * counts[right], dtype=np.float64)
        weights[diagonal] = counts[left[diagonal]] * (counts[left[diagonal]] - 1) / 2.0
        expected_pairs = instance_count * (instance_count - 1) // 2
        if not math.isclose(float(weights.sum()), float(expected_pairs)):
            raise RuntimeError("bootstrap pair weights do not reconstruct the sample")
        output[replicate] = _weighted_macro_f1(
            truth, model_prediction, weights
        ) - _weighted_macro_f1(truth, baseline_prediction, weights)
    return output


def evaluate_formal_tjdr_model_test(
    train_cache_path: str | Path,
    test_archives: Sequence[str | Path],
    config_path: str | Path,
    weights_directory: str | Path,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Evaluate the frozen selected three-seed ensemble on official test once."""

    config: Mapping[str, Any] = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seeds = tuple(int(value) for value in config["seeds"])
    _seed_everything(bootstrap_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal TJDR model evaluation requires CUDA")
    train_cache = load_tjdr_model_cache(train_cache_path)
    train_burdens = np.asarray(train_cache["burdens"], dtype=np.float32)
    train_partition = np.asarray(train_cache["partition"], dtype=np.uint8)
    train_indices = np.flatnonzero(train_partition == 0).astype(np.int64)
    relation_config = _relation_config(config)
    scales = _scales(config)
    train_targets = build_relation_targets(
        train_burdens,
        train_indices.tolist(),
        base_config=relation_config,
        scales=scales,
    )
    test_images, test_burdens, input_summary = load_tjdr_test_model_arrays(
        test_archives,
        expected_count=113,
        image_size=int(config["data"]["image_size"]),
    )
    test_indices = np.arange(test_images.shape[0], dtype=np.int64)
    off_diagonal_truth = build_relation_targets(
        test_burdens,
        test_indices.tolist(),
        base_config=relation_config,
        scales=scales,
    )
    weights_root = Path(weights_directory)
    burden_paths = tuple(weights_root / f"burden_encoder_seed{seed}.pt" for seed in seeds)
    head_paths = tuple(weights_root / f"rps_multitask_pair_seed{seed}.pt" for seed in seeds)
    embeddings = tuple(
        _image_embeddings(
            test_images,
            path,
            batch_size=int(config["burden_model"]["batch_size"]),
            device=device,
        )
        for path in burden_paths
    )
    left, right, truth, diagonal = _truth_with_diagonal(
        off_diagonal_truth, test_images.shape[0], len(scales)
    )
    prediction, transition_score, critical_prediction, per_seed_prediction = _pair_ensemble(
        embeddings,
        head_paths,
        left,
        right,
        scale_count=len(scales),
        batch_size=int(config["pair_model"]["batch_size"]),
        device=device,
    )
    off_prediction = prediction[~diagonal]
    baseline_off = majority_spectrum_baseline(train_targets, off_diagonal_truth.pair_count)
    majority_modes = baseline_off[0]
    baseline_all = np.tile(majority_modes, (truth.shape[0], 1))
    selected_macro_f1 = multiscale_macro_f1(
        off_diagonal_truth.relation_codes, off_prediction
    )
    baseline_macro_f1 = multiscale_macro_f1(
        off_diagonal_truth.relation_codes, baseline_off
    )
    selected_metrics = {
        "spectrum_macro_f1": selected_macro_f1,
        "relation_accuracy": relation_accuracy(
            off_diagonal_truth.relation_codes, off_prediction
        ),
        "phase_total_variation": phase_total_variation(
            off_diagonal_truth.relation_codes, off_prediction
        ),
        "transition_auc": binary_auc(
            off_diagonal_truth.transition, transition_score[~diagonal]
        ),
        "critical_log2_mae": critical_log2_mae(
            off_diagonal_truth.critical_log2, critical_prediction[~diagonal]
        ),
    }
    baseline_metrics = {
        "spectrum_macro_f1": baseline_macro_f1,
        "relation_accuracy": relation_accuracy(
            off_diagonal_truth.relation_codes, baseline_off
        ),
        "phase_total_variation": phase_total_variation(
            off_diagonal_truth.relation_codes, baseline_off
        ),
    }
    gains = paired_image_bootstrap_macro_f1_gain(
        truth,
        prediction,
        baseline_all,
        left,
        right,
        instance_count=test_images.shape[0],
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    lower, upper = np.quantile(gains, (0.025, 0.975), method="linear")
    per_seed_metrics = [
        {
            "seed": seed,
            "spectrum_macro_f1": multiscale_macro_f1(
                off_diagonal_truth.relation_codes, seed_prediction[~diagonal]
            ),
        }
        for seed, seed_prediction in zip(seeds, per_seed_prediction, strict=True)
    ]
    observed_gain = selected_macro_f1 - baseline_macro_f1
    return {
        "status": "RPS_MED_TJDR_MODEL_FORMAL_TEST_COMPLETE",
        "decision": "PASS" if float(lower) > 0.0 else "NO_GO",
        "selected_model": "rps_multitask_pair",
        "input": input_summary,
        "pair_count": off_diagonal_truth.pair_count,
        "selected_metrics": selected_metrics,
        "majority_baseline": baseline_metrics,
        "per_seed": per_seed_metrics,
        "primary": {
            "metric": "MULTISCALE_MACRO_F1_GAIN_OVER_SCALE_SPECIFIC_MAJORITY",
            "observed_gain": observed_gain,
            "paired_image_bootstrap_lower_95": float(lower),
            "paired_image_bootstrap_upper_95": float(upper),
            "success_threshold_strictly_above": 0.0,
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
        },
        "test_cache_persisted": False,
        "row_predictions_persisted": False,
        "source_identifiers_recorded": False,
        "model_retraining": False,
        "test_feedback_tuning": False,
    }


__all__ = [
    "evaluate_formal_tjdr_model_test",
    "paired_image_bootstrap_macro_f1_gain",
]

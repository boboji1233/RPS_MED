"""Development-only invariance and logical robustness checks for RPS-Med."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from numpy.typing import NDArray

from nts_med.relations import Relation
from rps_med.model_data import load_tjdr_model_cache
from rps_med.model_evaluation import _image_embeddings, _pair_ensemble
from rps_med.model_metrics import (
    binary_auc,
    build_relation_targets,
    multiscale_macro_f1,
    phase_total_variation,
    relation_accuracy,
)
from rps_med.model_training import _relation_config, _scales, _seed_everything


def orientation_reversal_consistency(
    forward: NDArray[np.int64], reverse: NDArray[np.int64]
) -> float:
    """Score the LESS↔GREATER and symmetric-class reversal contract."""

    if forward.shape != reverse.shape or forward.ndim != 2:
        raise ValueError("orientation relation arrays must align")
    mapping = np.asarray((1, 0, 2, 3, 4), dtype=np.int64)
    return float(np.mean(reverse == mapping[forward]))


def _transition_count_summary(values: NDArray[np.int64]) -> dict[str, Any]:
    counts = np.count_nonzero(values[:, 1:] != values[:, :-1], axis=1)
    distribution = Counter(int(value) for value in counts)
    return {
        "mean": float(np.mean(counts)),
        "maximum": int(np.max(counts)),
        "distribution": {str(key): distribution[key] for key in sorted(distribution)},
    }


def _pattern_coverage(
    train_truth: NDArray[np.int64], validation_prediction: NDArray[np.int64]
) -> float:
    known = {row.tobytes() for row in train_truth}
    return float(
        np.mean([row.tobytes() in known for row in validation_prediction])
    )


def _transforms(images: NDArray[np.uint8]) -> Mapping[str, NDArray[np.uint8]]:
    return {
        "horizontal_flip": np.flip(images, axis=2).copy(),
        "vertical_flip": np.flip(images, axis=1).copy(),
        "rotation_90": np.rot90(images, k=1, axes=(1, 2)).copy(),
    }


def audit_tjdr_model_robustness(
    cache_path: str | Path,
    config_path: str | Path,
    weights_directory: str | Path,
) -> dict[str, Any]:
    """Audit the selected ensemble only on the frozen development validation split."""

    cache = load_tjdr_model_cache(cache_path)
    config: Mapping[str, Any] = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    images = np.asarray(cache["images"], dtype=np.uint8)
    burdens = np.asarray(cache["burdens"], dtype=np.float32)
    partition = np.asarray(cache["partition"], dtype=np.uint8)
    train_indices = np.flatnonzero(partition == 0).astype(np.int64)
    validation_indices = np.flatnonzero(partition == 1).astype(np.int64)
    validation_images = images[validation_indices]
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
    burden_paths = tuple(weights_root / f"burden_encoder_seed{seed}.pt" for seed in seeds)
    head_paths = tuple(weights_root / f"rps_multitask_pair_seed{seed}.pt" for seed in seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("model robustness audit requires the frozen CUDA runtime")
    _seed_everything(20260830)
    batch_size = int(config["burden_model"]["batch_size"])
    pair_batch_size = int(config["pair_model"]["batch_size"])
    original_embeddings = tuple(
        _image_embeddings(
            validation_images, path, batch_size=batch_size, device=device
        )
        for path in burden_paths
    )
    left, right = np.triu_indices(validation_indices.size, k=1)
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    prediction, transition_score, critical_prediction, _ = _pair_ensemble(
        original_embeddings,
        head_paths,
        left,
        right,
        scale_count=len(scales),
        batch_size=pair_batch_size,
        device=device,
    )
    reverse_prediction, _, _, _ = _pair_ensemble(
        original_embeddings,
        head_paths,
        right,
        left,
        scale_count=len(scales),
        batch_size=pair_batch_size,
        device=device,
    )
    self_indices = np.arange(validation_indices.size, dtype=np.int64)
    self_prediction, _, _, _ = _pair_ensemble(
        original_embeddings,
        head_paths,
        self_indices,
        self_indices,
        scale_count=len(scales),
        batch_size=pair_batch_size,
        device=device,
    )
    equivalent_code = tuple(Relation).index(Relation.EQUIVALENT)
    transform_agreement: dict[str, float] = {}
    for name, transformed_images in _transforms(validation_images).items():
        transformed_embeddings = tuple(
            _image_embeddings(
                transformed_images, path, batch_size=batch_size, device=device
            )
            for path in burden_paths
        )
        transformed_prediction, _, _, _ = _pair_ensemble(
            transformed_embeddings,
            head_paths,
            left,
            right,
            scale_count=len(scales),
            batch_size=pair_batch_size,
            device=device,
        )
        transform_agreement[name] = float(np.mean(transformed_prediction == prediction))
    finite_critical = np.isfinite(validation_truth.critical_log2)
    critical_in_range = np.logical_and(critical_prediction >= -2.0, critical_prediction <= 2.0)
    transition_brier = float(
        np.mean(np.square(transition_score - validation_truth.transition))
    )
    return {
        "status": "RPS_MED_TJDR_MODEL_ROBUSTNESS_COMPLETE",
        "scope": "DEVELOPMENT_VALIDATION_ONLY",
        "validation_images": int(validation_indices.size),
        "validation_pairs": validation_truth.pair_count,
        "ensemble_metrics": {
            "spectrum_macro_f1": multiscale_macro_f1(
                validation_truth.relation_codes, prediction
            ),
            "relation_accuracy": relation_accuracy(
                validation_truth.relation_codes, prediction
            ),
            "phase_total_variation": phase_total_variation(
                validation_truth.relation_codes, prediction
            ),
            "transition_auc": binary_auc(
                validation_truth.transition, transition_score
            ),
            "transition_brier": transition_brier,
        },
        "logical_consistency": {
            "orientation_reversal_relation_consistency": orientation_reversal_consistency(
                prediction, reverse_prediction
            ),
            "self_pair_equivalent_fraction": float(
                np.mean(self_prediction == equivalent_code)
            ),
            "predicted_pattern_seen_in_train_fraction": _pattern_coverage(
                train_truth.relation_codes, prediction
            ),
            "predicted_transition_count": _transition_count_summary(prediction),
            "truth_transition_count": _transition_count_summary(
                validation_truth.relation_codes
            ),
            "critical_prediction_in_frozen_range_fraction_on_true_transitions": float(
                np.mean(critical_in_range[finite_critical])
            ),
        },
        "joint_transform_relation_agreement": transform_agreement,
        "official_test_accessed": False,
        "model_retraining": False,
        "row_predictions_persisted": False,
        "source_identifiers_recorded": False,
    }


__all__ = ["audit_tjdr_model_robustness", "orientation_reversal_consistency"]

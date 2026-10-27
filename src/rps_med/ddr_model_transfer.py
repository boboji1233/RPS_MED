"""Frozen TJDR-model transfer evaluation on DDR lesion official valid."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from numpy.typing import NDArray
from PIL import Image

from nts_med.concepts.fov import FOVStatus
from nts_med.data.adapters import AdapterStatus, DDRAdapter
from nts_med.gate0.source_io import _binary_mask, _burden, _disk_image
from rps_med.fov_v07 import extract_ddr_retinal_fov_v07
from rps_med.model_data import MODEL_AXES, load_tjdr_model_cache
from rps_med.model_evaluation import (
    _image_embeddings,
    _pair_ensemble,
    _truth_with_diagonal,
    paired_image_bootstrap_macro_f1_gain,
)
from rps_med.model_metrics import (
    binary_auc,
    build_relation_targets,
    critical_log2_mae,
    majority_spectrum_baseline,
    multiscale_macro_f1,
    phase_total_variation,
    relation_accuracy,
)
from rps_med.model_training import _relation_config, _scales, _seed_everything


def _resize_rgb(image: NDArray[Any], image_size: int) -> NDArray[np.uint8]:
    if image.ndim != 3 or image.shape[2] != 3 or image_size < 224 or image_size % 32:
        raise ValueError("DDR model image or resize contract is invalid")
    resized = Image.fromarray(np.asarray(image, dtype=np.uint8)).resize(
        (image_size, image_size), Image.Resampling.LANCZOS
    )
    return np.asarray(resized, dtype=np.uint8).copy()


def load_ddr_valid_model_arrays(
    root: str | Path,
    *,
    expected_count: int,
    image_size: int,
) -> tuple[NDArray[np.uint8], NDArray[np.float32], dict[str, Any]]:
    """Decode DDR valid images and frozen v0.7 burdens into memory only."""

    adapter = DDRAdapter(root)
    validation = adapter.validate()
    if validation.status is not AdapterStatus.READY:
        raise ValueError("DDR adapter must be ready")
    samples = tuple(sample for sample in adapter.samples() if sample.official_split == "valid")
    if len(samples) != expected_count:
        raise ValueError("DDR valid source boundary differs from the frozen count")
    images: list[NDArray[np.uint8]] = []
    burdens: list[list[float]] = []
    encoded_hashes: set[str] = set()
    fov_algorithms: Counter[str] = Counter()
    for sample in samples:
        encoded_hashes.add(hashlib.sha256(sample.image_path.read_bytes()).hexdigest())
        image = _disk_image(sample.image_path, rgb=True)
        fov = extract_ddr_retinal_fov_v07(image)
        if (
            fov.status is not FOVStatus.VALID
            or fov.mask is None
            or fov.area_ratio is None
            or not math.isfinite(fov.area_ratio)
        ):
            raise ValueError("DDR valid FOV violates the frozen v0.7 contract")
        masks = {
            axis: _binary_mask(
                _disk_image(sample.lesion_masks[axis], rgb=False),
                positive_class=255,
            )
            for axis in MODEL_AXES
        }
        burden = _burden(masks, fov.mask)
        images.append(_resize_rgb(image, image_size))
        burdens.append([float(burden[axis]) for axis in MODEL_AXES])
        fov_algorithms[fov.algorithm_version] += 1
    if len(encoded_hashes) != expected_count:
        raise ValueError("DDR valid contains duplicate encoded image content")
    image_array = np.stack(images)
    burden_array = np.asarray(burdens, dtype=np.float32)
    if image_array.shape != (expected_count, image_size, image_size, 3):
        raise ValueError("DDR model image array shape is invalid")
    if burden_array.shape != (expected_count, len(MODEL_AXES)):
        raise ValueError("DDR burden array shape is invalid")
    return (
        image_array,
        burden_array,
        {
            "input_image_count": len(samples),
            "unique_encoded_image_count": len(encoded_hashes),
            "measurement_eligible_count": len(images),
            "measurement_exclusion_count": 0,
            "fov_algorithm_counts": dict(sorted(fov_algorithms.items())),
            "image_size": image_size,
            "image_cache_persisted": False,
            "source_identifiers_recorded": False,
            "source_paths_recorded": False,
        },
    )


def evaluate_ddr_model_transfer(
    train_cache_path: str | Path,
    ddr_root: str | Path,
    config_path: str | Path,
    weights_directory: str | Path,
    *,
    expected_count: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Evaluate the already selected ensemble without model or seed reselection."""

    config: Mapping[str, Any] = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seeds = tuple(int(value) for value in config["seeds"])
    if seeds != (20260826, 20260827, 20260828):
        raise ValueError("DDR transfer seed ensemble differs from the selected model")
    _seed_everything(bootstrap_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DDR model transfer requires the frozen CUDA runtime")
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
    images, burdens, input_summary = load_ddr_valid_model_arrays(
        ddr_root,
        expected_count=expected_count,
        image_size=int(config["data"]["image_size"]),
    )
    ddr_indices = np.arange(expected_count, dtype=np.int64)
    off_truth = build_relation_targets(
        burdens,
        ddr_indices.tolist(),
        base_config=relation_config,
        scales=scales,
    )
    weights_root = Path(weights_directory)
    burden_paths = tuple(weights_root / f"burden_encoder_seed{seed}.pt" for seed in seeds)
    head_paths = tuple(weights_root / f"rps_multitask_pair_seed{seed}.pt" for seed in seeds)
    embeddings = tuple(
        _image_embeddings(
            images,
            path,
            batch_size=int(config["burden_model"]["batch_size"]),
            device=device,
        )
        for path in burden_paths
    )
    left, right, truth, diagonal = _truth_with_diagonal(
        off_truth, expected_count, len(scales)
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
    baseline_off = majority_spectrum_baseline(train_targets, off_truth.pair_count)
    baseline_all = np.tile(baseline_off[0], (truth.shape[0], 1))
    selected_macro_f1 = multiscale_macro_f1(off_truth.relation_codes, off_prediction)
    baseline_macro_f1 = multiscale_macro_f1(off_truth.relation_codes, baseline_off)
    gains = paired_image_bootstrap_macro_f1_gain(
        truth,
        prediction,
        baseline_all,
        left,
        right,
        instance_count=expected_count,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    lower, upper = np.quantile(gains, (0.025, 0.975))
    passed = float(lower) > 0.0
    return {
        "status": "RPS_MED_DDR_MODEL_TRANSFER_COMPLETE",
        "decision": "TRANSFER_PASS" if passed else "TRANSFER_NOT_CONFIRMED",
        "analysis_role": "EXTERNAL_TRANSFER_SECONDARY_CONFIRMATION",
        "prior_source_exposure": "MODEL_FREE_RELATION_ANALYSIS_ONLY",
        "input_summary": input_summary,
        "pair_count": off_truth.pair_count,
        "primary": {
            "metric": "MULTISCALE_MACRO_F1_GAIN_SELECTED_MINUS_TJDR_TRAIN_MAJORITY",
            "selected": selected_macro_f1,
            "baseline": baseline_macro_f1,
            "observed_gain": selected_macro_f1 - baseline_macro_f1,
            "bootstrap_lower_95": float(lower),
            "bootstrap_upper_95": float(upper),
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "pass_rule": "LOWER_95_STRICTLY_ABOVE_ZERO",
            "pass": passed,
        },
        "selected_secondary": {
            "relation_accuracy": relation_accuracy(off_truth.relation_codes, off_prediction),
            "phase_total_variation": phase_total_variation(
                off_truth.relation_codes, off_prediction
            ),
            "transition_auc": binary_auc(
                off_truth.transition, transition_score[~diagonal]
            ),
            "critical_log2_mae": critical_log2_mae(
                off_truth.critical_log2, critical_prediction[~diagonal]
            ),
            "per_seed_macro_f1": {
                str(seed): multiscale_macro_f1(
                    off_truth.relation_codes, seed_prediction[~diagonal]
                )
                for seed, seed_prediction in zip(
                    seeds, per_seed_prediction, strict=True
                )
            },
        },
        "baseline_secondary": {
            "relation_accuracy": relation_accuracy(off_truth.relation_codes, baseline_off),
            "phase_total_variation": phase_total_variation(
                off_truth.relation_codes, baseline_off
            ),
        },
        "image_cache_persisted": False,
        "row_predictions_persisted": False,
        "source_identifiers_recorded": False,
        "model_retraining": False,
        "model_or_seed_reselection": False,
        "result_feedback_tuning": False,
        "prior_formal_results_changed": False,
    }


__all__ = ["evaluate_ddr_model_transfer", "load_ddr_valid_model_arrays"]

"""Development-only training workflow for RPS-Med TJDR image models."""

from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# PyTorch requires this before the first CUDA context is created when
# deterministic CuBLAS kernels are requested.  It changes execution mechanics,
# not the frozen scientific or optimization contract.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as functional
import yaml
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from nts_med.relations import Relation, RelationConfig
from nts_med.reproducibility.freeze import sha256_file
from rps_med.model_data import load_tjdr_model_cache
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
from rps_med.modeling import BurdenEncoder, PairSpectrumHead, load_resnet18_imagenet_v1

_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


class _ImageDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    def __init__(
        self,
        images: NDArray[np.uint8],
        burdens: NDArray[np.float32],
        presence: NDArray[np.float32],
        indices: NDArray[np.int64],
        *,
        augment: bool,
    ) -> None:
        self.images = images
        self.burdens = burdens
        self.presence = presence
        self.indices = indices
        self.augment = augment

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        index = int(self.indices[item])
        image = torch.from_numpy(self.images[index].copy()).permute(2, 0, 1).float() / 255.0
        if self.augment:
            if bool(torch.rand(()) < 0.5):
                image = torch.flip(image, dims=(2,))
            if bool(torch.rand(()) < 0.5):
                image = torch.flip(image, dims=(1,))
            image = torch.rot90(image, int(torch.randint(0, 4, ()).item()), dims=(1, 2))
            brightness = 0.9 + 0.2 * float(torch.rand(()).item())
            contrast = 0.9 + 0.2 * float(torch.rand(()).item())
            saturation = 0.9 + 0.2 * float(torch.rand(()).item())
            image = torch.clamp(image * brightness, 0.0, 1.0)
            channel_mean = image.mean(dim=(1, 2), keepdim=True)
            image = torch.clamp((image - channel_mean) * contrast + channel_mean, 0.0, 1.0)
            gray = image.mean(dim=0, keepdim=True)
            image = torch.clamp((image - gray) * saturation + gray, 0.0, 1.0)
        image = (image - _MEAN) / _STD
        return (
            image,
            torch.from_numpy(self.burdens[index]),
            torch.from_numpy(self.presence[index]),
            torch.tensor(index, dtype=torch.int64),
        )


@dataclass(frozen=True, slots=True)
class _ModelMetrics:
    spectrum_macro_f1: float
    relation_accuracy: float
    phase_total_variation: float
    transition_auc: float | None
    critical_log2_mae: float | None


def _seed_everything(seed: int) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("deterministic CuBLAS workspace is not configured")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _relation_config(config: Mapping[str, Any]) -> RelationConfig:
    thresholds = config["relation"]["thresholds"]
    return RelationConfig(
        delta=thresholds["delta"],
        epsilon=thresholds["epsilon"],
        max_width=thresholds["max_width"],
        numeric_tolerance=float(thresholds["numeric_tolerance"]),
    )


def _scales(config: Mapping[str, Any]) -> tuple[float, ...]:
    values = config["relation"]["scales"]
    scales = tuple(float(value) for value in values)
    if len(scales) != 17 or any(value <= 0.0 for value in scales):
        raise ValueError("model protocol requires the frozen 17-point positive scale grid")
    return scales


def _loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def _evaluate_burden_model(
    model: BurdenEncoder,
    loader: DataLoader[Any],
    *,
    device: torch.device,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]:
    model.eval()
    burden_rows: list[NDArray[np.float32]] = []
    presence_rows: list[NDArray[np.float32]] = []
    embedding_rows: list[NDArray[np.float32]] = []
    indices: list[NDArray[np.int64]] = []
    with torch.inference_mode():
        for images, _, _, batch_indices in loader:
            embedding, burden, presence = model(images.to(device, non_blocking=True))
            embedding_rows.append(embedding.float().cpu().numpy())
            burden_rows.append(burden.float().cpu().numpy())
            presence_rows.append(torch.sigmoid(presence).float().cpu().numpy())
            indices.append(batch_indices.numpy())
    return (
        np.concatenate(burden_rows),
        np.concatenate(presence_rows),
        np.concatenate(embedding_rows),
        np.concatenate(indices),
    )


def _spectrum_metrics(
    truth: PairTargets,
    prediction: NDArray[np.int64],
    *,
    transition_score: NDArray[np.float32] | None = None,
    critical_prediction: NDArray[np.float32] | None = None,
) -> _ModelMetrics:
    return _ModelMetrics(
        spectrum_macro_f1=multiscale_macro_f1(truth.relation_codes, prediction),
        relation_accuracy=relation_accuracy(truth.relation_codes, prediction),
        phase_total_variation=phase_total_variation(truth.relation_codes, prediction),
        transition_auc=(
            None if transition_score is None else binary_auc(truth.transition, transition_score)
        ),
        critical_log2_mae=(
            None
            if critical_prediction is None
            else critical_log2_mae(truth.critical_log2, critical_prediction)
        ),
    )


def _analytic_prediction_metrics(
    predicted_burdens: NDArray[np.float32],
    indices: NDArray[np.int64],
    truth: PairTargets,
    *,
    relation_config: RelationConfig,
    scales: Sequence[float],
) -> _ModelMetrics:
    predicted = build_relation_targets(
        predicted_burdens,
        indices.tolist(),
        base_config=relation_config,
        scales=scales,
    )
    critical = predicted.critical_log2.copy()
    missing = np.logical_and(np.isfinite(truth.critical_log2), ~np.isfinite(critical))
    critical[missing] = 0.0
    return _spectrum_metrics(
        truth,
        predicted.relation_codes,
        transition_score=predicted.transition,
        critical_prediction=critical,
    )


def _train_burden_encoder(
    images: NDArray[np.uint8],
    burdens: NDArray[np.float32],
    presence: NDArray[np.float32],
    train_indices: NDArray[np.int64],
    validation_indices: NDArray[np.int64],
    validation_targets: PairTargets,
    *,
    relation_config: RelationConfig,
    scales: Sequence[float],
    weights_path: Path,
    output_path: Path,
    seed: int,
    device: torch.device,
    config: Mapping[str, Any],
) -> tuple[BurdenEncoder, _ModelMetrics, NDArray[np.float32]]:
    settings = config["burden_model"]
    _seed_everything(seed)
    model = BurdenEncoder()
    load_resnet18_imagenet_v1(model, weights_path)
    model.to(device)
    target_mean = burdens[train_indices].mean(axis=0).astype(np.float32)
    target_std = burdens[train_indices].std(axis=0).astype(np.float32)
    target_std = np.maximum(target_std, np.float32(1.0e-6))
    mean_tensor = torch.from_numpy(target_mean).to(device)
    std_tensor = torch.from_numpy(target_std).to(device)
    train_loader = _loader(
        _ImageDataset(images, burdens, presence, train_indices, augment=True),
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        seed=seed,
        device=device,
    )
    validation_loader = _loader(
        _ImageDataset(images, burdens, presence, validation_indices, augment=False),
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        seed=seed,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": float(settings["encoder_lr"])},
            {
                "params": (*model.burden_head.parameters(), *model.presence_head.parameters()),
                "lr": float(settings["head_lr"]),
            },
        ],
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(settings["max_epochs"])
    )
    use_amp = bool(settings["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # type: ignore[attr-defined]
    best_score = -math.inf
    stale_epochs = 0
    best_metrics: _ModelMetrics | None = None
    for epoch in range(1, int(settings["max_epochs"]) + 1):
        model.train()
        for batch_images, batch_burdens, batch_presence, _ in train_loader:
            batch_images = batch_images.to(device, non_blocking=True)
            batch_burdens = batch_burdens.to(device, non_blocking=True)
            batch_presence = batch_presence.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(  # type: ignore[attr-defined]
                device.type, enabled=use_amp
            ):
                _, burden_prediction, presence_logits = model(batch_images)
                standardized_prediction = (burden_prediction - mean_tensor) / std_tensor
                standardized_truth = (batch_burdens - mean_tensor) / std_tensor
                burden_loss = functional.smooth_l1_loss(
                    standardized_prediction,
                    standardized_truth,
                    beta=float(settings["huber_beta"]),
                )
                presence_loss = functional.binary_cross_entropy_with_logits(
                    presence_logits, batch_presence
                )
                loss = burden_loss + float(settings["presence_loss_weight"]) * presence_loss
            scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        predicted_burdens, _, _, observed_indices = _evaluate_burden_model(
            model, validation_loader, device=device
        )
        ordered = np.argsort(observed_indices)
        predicted_full = np.zeros_like(burdens)
        predicted_full[observed_indices[ordered]] = predicted_burdens[ordered]
        metrics = _analytic_prediction_metrics(
            predicted_full,
            validation_indices,
            validation_targets,
            relation_config=relation_config,
            scales=scales,
        )
        score = metrics.spectrum_macro_f1
        print(
            json.dumps(
                {
                    "stage": "BURDEN_ENCODER",
                    "seed": seed,
                    "epoch": epoch,
                    "validation_macro_f1": score,
                }
            ),
            flush=True,
        )
        if score > best_score + float(settings["early_stopping_min_delta"]):
            best_score = score
            best_metrics = metrics
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "seed": seed,
                    "epoch": epoch,
                },
                output_path,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= int(settings["early_stopping_patience"]):
            break
    if best_metrics is None:
        raise RuntimeError("burden encoder did not produce a finite validation metric")
    checkpoint = torch.load(output_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    all_loader = _loader(
        _ImageDataset(
            images,
            burdens,
            presence,
            np.arange(images.shape[0], dtype=np.int64),
            augment=False,
        ),
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        seed=seed,
        device=device,
    )
    predicted_burdens, _, embeddings, observed_indices = _evaluate_burden_model(
        model, all_loader, device=device
    )
    order = np.argsort(observed_indices)
    return model, best_metrics, embeddings[order]


def _class_weights(targets: PairTargets, *, maximum: float) -> Tensor:
    counts = np.bincount(targets.relation_codes.ravel(), minlength=len(Relation)).astype(
        np.float64
    )
    weights = np.zeros(len(Relation), dtype=np.float64)
    nonzero = counts > 0
    weights[nonzero] = 1.0 / np.sqrt(counts[nonzero])
    weights[nonzero] /= weights[nonzero].mean()
    weights = np.minimum(weights, maximum)
    return torch.from_numpy(weights.astype(np.float32))


def _pair_dataset(targets: PairTargets) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(targets.left_indices),
        torch.from_numpy(targets.right_indices),
        torch.from_numpy(targets.relation_codes),
        torch.from_numpy(targets.transition),
        torch.from_numpy(targets.critical_log2),
    )


def _pair_predictions(
    head: PairSpectrumHead,
    embeddings: Tensor,
    targets: PairTargets,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.float32]]:
    loader = _loader(
        _pair_dataset(targets),
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        device=device,
    )
    predictions: list[NDArray[np.int64]] = []
    transitions: list[NDArray[np.float32]] = []
    critical: list[NDArray[np.float32]] = []
    head.eval()
    with torch.inference_mode():
        for left, right, _, _, _ in loader:
            logits, transition, critical_log2 = head(
                embeddings[left.to(device)], embeddings[right.to(device)]
            )
            predictions.append(logits.argmax(dim=2).cpu().numpy())
            transitions.append(torch.sigmoid(transition).float().cpu().numpy())
            critical.append(critical_log2.float().cpu().numpy())
    return (
        np.concatenate(predictions),
        np.concatenate(transitions),
        np.concatenate(critical),
    )


def _train_pair_head(
    embeddings_array: NDArray[np.float32],
    train_targets: PairTargets,
    validation_targets: PairTargets,
    *,
    output_path: Path,
    seed: int,
    device: torch.device,
    config: Mapping[str, Any],
    stage: str,
) -> _ModelMetrics:
    settings = config["pair_model"]
    _seed_everything(seed)
    embeddings = torch.from_numpy(embeddings_array).to(device)
    head = PairSpectrumHead(
        scale_count=train_targets.relation_codes.shape[1],
        relation_count=len(Relation),
        dropout=float(settings["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(settings["max_epochs"])
    )
    train_loader = _loader(
        _pair_dataset(train_targets),
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        seed=seed,
        device=device,
    )
    class_weights = _class_weights(
        train_targets, maximum=float(settings["maximum_class_weight"])
    ).to(device)
    finite_critical = np.isfinite(train_targets.critical_log2)
    critical_mean = float(np.mean(train_targets.critical_log2[finite_critical]))
    critical_std = max(float(np.std(train_targets.critical_log2[finite_critical])), 1.0e-6)
    use_amp = bool(settings["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # type: ignore[attr-defined]
    best_score = -math.inf
    best_metrics: _ModelMetrics | None = None
    stale_epochs = 0
    for epoch in range(1, int(settings["max_epochs"]) + 1):
        head.train()
        for left, right, relation, transition, critical in train_loader:
            left = left.to(device)
            right = right.to(device)
            relation = relation.to(device)
            transition = transition.to(device)
            critical = critical.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(  # type: ignore[attr-defined]
                device.type, enabled=use_amp
            ):
                logits, transition_logit, critical_prediction = head(
                    embeddings[left], embeddings[right]
                )
                spectrum_loss = functional.cross_entropy(
                    logits.reshape(-1, len(Relation)),
                    relation.reshape(-1),
                    weight=class_weights,
                )
                transition_loss = functional.binary_cross_entropy_with_logits(
                    transition_logit, transition
                )
                transition_selector = transition > 0.5
                if bool(transition_selector.any()):
                    standardized_critical = (
                        critical[transition_selector] - critical_mean
                    ) / critical_std
                    critical_loss = functional.smooth_l1_loss(
                        critical_prediction[transition_selector],
                        standardized_critical,
                        beta=float(settings["critical_huber_beta"]),
                    )
                else:
                    critical_loss = spectrum_loss.new_zeros(())
                loss = (
                    spectrum_loss
                    + float(settings["transition_loss_weight"]) * transition_loss
                    + float(settings["critical_loss_weight"]) * critical_loss
                )
            scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        prediction, transition_score, critical_standardized = _pair_predictions(
            head,
            embeddings,
            validation_targets,
            batch_size=int(settings["batch_size"]),
            seed=seed,
            device=device,
        )
        critical_prediction = critical_standardized * critical_std + critical_mean
        metrics = _spectrum_metrics(
            validation_targets,
            prediction,
            transition_score=transition_score,
            critical_prediction=critical_prediction,
        )
        score = metrics.spectrum_macro_f1
        print(
            json.dumps(
                {
                    "stage": stage,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_macro_f1": score,
                }
            ),
            flush=True,
        )
        if score > best_score + float(settings["early_stopping_min_delta"]):
            best_score = score
            best_metrics = metrics
            stale_epochs = 0
            torch.save(
                {
                    "model": head.state_dict(),
                    "critical_mean": critical_mean,
                    "critical_std": critical_std,
                    "seed": seed,
                    "epoch": epoch,
                },
                output_path,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= int(settings["early_stopping_patience"]):
            break
    if best_metrics is None:
        raise RuntimeError("pair head did not produce a finite validation metric")
    return best_metrics


def _frozen_imagenet_embeddings(
    images: NDArray[np.uint8],
    burdens: NDArray[np.float32],
    presence: NDArray[np.float32],
    *,
    weights_path: Path,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> NDArray[np.float32]:
    _seed_everything(seed)
    model = BurdenEncoder()
    load_resnet18_imagenet_v1(model, weights_path)
    model.to(device)
    loader = _loader(
        _ImageDataset(
            images,
            burdens,
            presence,
            np.arange(images.shape[0], dtype=np.int64),
            augment=False,
        ),
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        device=device,
    )
    _, _, embeddings, indices = _evaluate_burden_model(model, loader, device=device)
    return embeddings[np.argsort(indices)]


def _aggregate_seed_metrics(
    per_seed: Sequence[Mapping[str, Any]], model_name: str
) -> dict[str, float]:
    values = np.asarray(
        [row[model_name]["spectrum_macro_f1"] for row in per_seed], dtype=np.float64
    )
    return {
        "mean_spectrum_macro_f1": float(values.mean()),
        "standard_deviation_spectrum_macro_f1": (
            0.0 if values.size == 1 else float(values.std(ddof=1))
        ),
        "minimum_seed_spectrum_macro_f1": float(values.min()),
        "maximum_seed_spectrum_macro_f1": float(values.max()),
    }


def run_tjdr_model_development(
    cache_path: str | Path,
    pretrained_weights: str | Path,
    config_path: str | Path,
    weights_directory: str | Path,
) -> dict[str, Any]:
    """Train all preregistered development candidates without official-test access."""

    cache = load_tjdr_model_cache(cache_path)
    config: Mapping[str, Any] = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    images = np.asarray(cache["images"], dtype=np.uint8)
    burdens = np.asarray(cache["burdens"], dtype=np.float32)
    presence = np.asarray(cache["presence"], dtype=np.float32)
    partition = np.asarray(cache["partition"], dtype=np.uint8)
    train_indices = np.flatnonzero(partition == 0).astype(np.int64)
    validation_indices = np.flatnonzero(partition == 1).astype(np.int64)
    relation_config = _relation_config(config)
    scales = _scales(config)
    train_targets = build_relation_targets(
        burdens, train_indices.tolist(), base_config=relation_config, scales=scales
    )
    validation_targets = build_relation_targets(
        burdens,
        validation_indices.tolist(),
        base_config=relation_config,
        scales=scales,
    )
    majority_prediction = majority_spectrum_baseline(
        train_targets, validation_targets.pair_count
    )
    majority_metrics = _spectrum_metrics(validation_targets, majority_prediction)
    weights_root = Path(weights_directory)
    weights_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["runtime"]["device"] == "cuda" else "cpu"
    )
    if config["runtime"]["device"] == "cuda" and device.type != "cuda":
        raise RuntimeError("frozen model protocol requires CUDA but CUDA is unavailable")
    per_seed: list[dict[str, Any]] = []
    seeds = tuple(int(value) for value in config["seeds"])
    for seed in seeds:
        burden_path = weights_root / f"burden_encoder_seed{seed}.pt"
        frozen_pair_path = weights_root / f"frozen_imagenet_pair_seed{seed}.pt"
        proposed_pair_path = weights_root / f"rps_multitask_pair_seed{seed}.pt"
        for path in (burden_path, frozen_pair_path, proposed_pair_path):
            if path.exists():
                raise FileExistsError("development model weight output already exists")
        _, burden_metrics, learned_embeddings = _train_burden_encoder(
            images,
            burdens,
            presence,
            train_indices,
            validation_indices,
            validation_targets,
            relation_config=relation_config,
            scales=scales,
            weights_path=Path(pretrained_weights),
            output_path=burden_path,
            seed=seed,
            device=device,
            config=config,
        )
        frozen_embeddings = _frozen_imagenet_embeddings(
            images,
            burdens,
            presence,
            weights_path=Path(pretrained_weights),
            batch_size=int(config["burden_model"]["batch_size"]),
            seed=seed,
            device=device,
        )
        frozen_metrics = _train_pair_head(
            frozen_embeddings,
            train_targets,
            validation_targets,
            output_path=frozen_pair_path,
            seed=seed,
            device=device,
            config=config,
            stage="FROZEN_IMAGENET_PAIR_HEAD",
        )
        proposed_metrics = _train_pair_head(
            learned_embeddings,
            train_targets,
            validation_targets,
            output_path=proposed_pair_path,
            seed=seed,
            device=device,
            config=config,
            stage="RPS_MULTITASK_PAIR_HEAD",
        )
        per_seed.append(
            {
                "seed": seed,
                "image_burden_analytic": asdict(burden_metrics),
                "frozen_imagenet_pair": asdict(frozen_metrics),
                "rps_multitask_pair": asdict(proposed_metrics),
                "weight_sha256": {
                    "image_burden_analytic": sha256_file(burden_path),
                    "frozen_imagenet_pair": sha256_file(frozen_pair_path),
                    "rps_multitask_pair": sha256_file(proposed_pair_path),
                },
            }
        )
    model_names = (
        "image_burden_analytic",
        "frozen_imagenet_pair",
        "rps_multitask_pair",
    )
    aggregate = {name: _aggregate_seed_metrics(per_seed, name) for name in model_names}
    selected = max(
        model_names,
        key=lambda name: (aggregate[name]["mean_spectrum_macro_f1"], name),
    )
    margin = float(config["development_gate"]["minimum_macro_f1_margin_over_majority"])
    seed_pass = all(
        float(row[selected]["spectrum_macro_f1"])
        >= majority_metrics.spectrum_macro_f1 + margin
        for row in per_seed
    )
    mean_pass = (
        aggregate[selected]["mean_spectrum_macro_f1"]
        >= majority_metrics.spectrum_macro_f1 + margin
    )
    return {
        "status": "RPS_MED_TJDR_MODEL_DEVELOPMENT_COMPLETE",
        "decision": "PASS" if mean_pass and seed_pass else "NO_GO",
        "cache_sha256": sha256_file(Path(cache_path)),
        "config_sha256": sha256_file(Path(config_path)),
        "pretrained_weights_sha256": sha256_file(Path(pretrained_weights)),
        "runtime": {
            "python": str(torch.__config__.show()).splitlines()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        },
        "development": {
            "unique_image_count": int(images.shape[0]),
            "train_image_count": int(train_indices.size),
            "validation_image_count": int(validation_indices.size),
            "train_pair_count": train_targets.pair_count,
            "validation_pair_count": validation_targets.pair_count,
        },
        "majority_baseline": asdict(majority_metrics),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "selection": {
            "selected_model": selected,
            "criterion": "HIGHEST_MEAN_VALIDATION_MULTISCALE_MACRO_F1",
            "minimum_margin_over_majority": margin,
            "mean_margin_pass": mean_pass,
            "every_seed_margin_pass": seed_pass,
        },
        "official_test_accessed": False,
        "official_test_used_for_model_selection": False,
        "source_identifiers_recorded": False,
        "model_weights_git_ignored": True,
    }


def canonical_result_json(result: Mapping[str, Any]) -> str:
    return json.dumps(dict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


__all__ = ["canonical_result_json", "run_tjdr_model_development"]

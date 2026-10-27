"""Append-only development execution for RPS-Med Stage 3A."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as functional
import yaml
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from nts_med.reproducibility.freeze import sha256_file
from rps_med.model_data import load_tjdr_model_cache
from rps_med.model_metrics import (
    PairTargets,
    build_relation_targets,
    majority_spectrum_baseline,
)
from rps_med.model_training import (
    _analytic_prediction_metrics,
    _class_weights,
    _loader,
    _relation_config,
    _scales,
    _seed_everything,
)
from rps_med.modeling import BurdenEncoder, load_resnet18_imagenet_v1
from rps_med.stage3a_models import (
    ConfigurablePairHead,
    PairRepresentation,
    RankNetScalarHead,
    Stage3AMetrics,
    evaluate_stage3a_predictions,
    parameter_count,
    soft_admissibility_penalty,
)

_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
_INVERSION = np.asarray((1, 0, 2, 3, 4), dtype=np.int64)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError("Stage 3A append-only JSON already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_npz_once(path: Path, **arrays: NDArray[Any]) -> None:
    if path.exists():
        raise FileExistsError("Stage 3A append-only prediction artifact already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _git_commit(repository_root: Path) -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _require_tracked_clean(repository_root: Path) -> None:
    completed = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip():
        raise RuntimeError("Stage 3A requires a clean tracked worktree before training")


def _device_from_config(config: Mapping[str, Any]) -> torch.device:
    required = str(config["runtime"]["device"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if required == "cuda" and device.type != "cuda":
        raise RuntimeError("Stage 3A frozen protocol requires CUDA")
    return device


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    runtime_seconds: float
    peak_cuda_vram_bytes: int


def _resource_start(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def _resource_end(start: float, device: torch.device) -> ResourceUsage:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = 0
    return ResourceUsage(runtime_seconds=time.perf_counter() - start, peak_cuda_vram_bytes=peak)


class _Stage3AImageDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    def __init__(
        self,
        images: NDArray[np.uint8],
        burdens: NDArray[np.float32],
        presence: NDArray[np.float32],
        indices: NDArray[np.int64],
        *,
        training: bool,
        geometric_augmentation: bool,
    ) -> None:
        self.images = images
        self.burdens = burdens
        self.presence = presence
        self.indices = indices
        self.training = training
        self.geometric_augmentation = geometric_augmentation

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        index = int(self.indices[item])
        image = torch.from_numpy(self.images[index].copy()).permute(2, 0, 1).float() / 255.0
        if self.training and self.geometric_augmentation:
            if bool(torch.rand(()) < 0.5):
                image = torch.flip(image, dims=(2,))
            if bool(torch.rand(()) < 0.5):
                image = torch.flip(image, dims=(1,))
            image = torch.rot90(image, int(torch.randint(0, 4, ()).item()), dims=(1, 2))
        if self.training:
            brightness = 0.9 + 0.2 * float(torch.rand(()).item())
            contrast = 0.9 + 0.2 * float(torch.rand(()).item())
            saturation = 0.9 + 0.2 * float(torch.rand(()).item())
            image = torch.clamp(image * brightness, 0.0, 1.0)
            channel_mean = image.mean(dim=(1, 2), keepdim=True)
            image = torch.clamp((image - channel_mean) * contrast + channel_mean, 0.0, 1.0)
            gray = image.mean(dim=0, keepdim=True)
            image = torch.clamp((image - gray) * saturation + gray, 0.0, 1.0)
        return (
            (image - _MEAN) / _STD,
            torch.from_numpy(self.burdens[index]),
            torch.from_numpy(self.presence[index]),
            torch.tensor(index, dtype=torch.int64),
        )


def _evaluate_encoder(
    model: BurdenEncoder,
    loader: DataLoader[Any],
    *,
    device: torch.device,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.int64],
]:
    model.eval()
    burden_rows: list[NDArray[np.float32]] = []
    presence_rows: list[NDArray[np.float32]] = []
    embedding_rows: list[NDArray[np.float32]] = []
    index_rows: list[NDArray[np.int64]] = []
    with torch.inference_mode():
        for images, _, _, indices in loader:
            embedding, burden, presence = model(images.to(device, non_blocking=True))
            embedding_rows.append(embedding.float().cpu().numpy())
            burden_rows.append(burden.float().cpu().numpy())
            presence_rows.append(presence.float().cpu().numpy())
            index_rows.append(indices.numpy())
    return (
        np.concatenate(burden_rows),
        np.concatenate(presence_rows),
        np.concatenate(embedding_rows),
        np.concatenate(index_rows),
    )


@dataclass(frozen=True, slots=True)
class EncoderArtifact:
    contract: str
    seed: int
    embeddings: NDArray[np.float32]
    predicted_burdens: NDArray[np.float32] | None
    best_epoch: int | None
    validation_score: float | None
    resource: ResourceUsage
    parameter_count: int
    trainable_parameter_count: int
    checkpoint_sha256: str | None


def _frozen_encoder_embeddings(
    images: NDArray[np.uint8],
    burdens: NDArray[np.float32],
    presence: NDArray[np.float32],
    *,
    weights_path: Path,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> EncoderArtifact:
    _seed_everything(seed)
    model = BurdenEncoder()
    load_resnet18_imagenet_v1(model, weights_path)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.to(device)
    loader = _loader(
        _Stage3AImageDataset(
            images,
            burdens,
            presence,
            np.arange(images.shape[0], dtype=np.int64),
            training=False,
            geometric_augmentation=False,
        ),
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        device=device,
    )
    started = _resource_start(device)
    _, _, embeddings, indices = _evaluate_encoder(model, loader, device=device)
    resource = _resource_end(started, device)
    return EncoderArtifact(
        contract="frozen_imagenet",
        seed=seed,
        embeddings=embeddings[np.argsort(indices)],
        predicted_burdens=None,
        best_epoch=None,
        validation_score=None,
        resource=resource,
        parameter_count=parameter_count(model),
        trainable_parameter_count=0,
        checkpoint_sha256=None,
    )


def _train_encoder_contract(
    contract: str,
    spec: Mapping[str, Any],
    images: NDArray[np.uint8],
    burdens: NDArray[np.float32],
    presence: NDArray[np.float32],
    train_indices: NDArray[np.int64],
    validation_indices: NDArray[np.int64],
    validation_targets: PairTargets,
    *,
    relation_config: Any,
    scales: Sequence[float],
    weights_path: Path,
    output_directory: Path,
    seed: int,
    device: torch.device,
    common: Mapping[str, Any],
) -> EncoderArtifact:
    _seed_everything(seed)
    settings = common["encoder_training"]
    burden_weight = float(spec["burden_weight"])
    presence_weight = float(spec["presence_weight"])
    if burden_weight <= 0.0 and presence_weight <= 0.0:
        raise ValueError("trainable Stage 3A encoder requires a nonzero supervision weight")
    model = BurdenEncoder()
    load_resnet18_imagenet_v1(model, weights_path)
    model.to(device)
    target_mean = burdens[train_indices].mean(axis=0).astype(np.float32)
    target_std = np.maximum(
        burdens[train_indices].std(axis=0).astype(np.float32), np.float32(1.0e-6)
    )
    mean_tensor = torch.from_numpy(target_mean).to(device)
    std_tensor = torch.from_numpy(target_std).to(device)
    train_loader = _loader(
        _Stage3AImageDataset(
            images,
            burdens,
            presence,
            train_indices,
            training=True,
            geometric_augmentation=bool(spec["geometric_augmentation"]),
        ),
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        seed=seed,
        device=device,
    )
    validation_loader = _loader(
        _Stage3AImageDataset(
            images,
            burdens,
            presence,
            validation_indices,
            training=False,
            geometric_augmentation=False,
        ),
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
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    started = _resource_start(device)
    for epoch in range(1, int(settings["max_epochs"]) + 1):
        model.train()
        for batch_images, batch_burdens, batch_presence, _ in train_loader:
            batch_images = batch_images.to(device, non_blocking=True)
            batch_burdens = batch_burdens.to(device, non_blocking=True)
            batch_presence = batch_presence.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):  # type: ignore[attr-defined]
                _, burden_prediction, presence_logits = model(batch_images)
                loss = burden_prediction.new_zeros(())
                if burden_weight > 0.0:
                    burden_loss = functional.smooth_l1_loss(
                        (burden_prediction - mean_tensor) / std_tensor,
                        (batch_burdens - mean_tensor) / std_tensor,
                        beta=float(settings["burden_huber_beta"]),
                    )
                    loss = loss + burden_weight * burden_loss
                if presence_weight > 0.0:
                    presence_loss = functional.binary_cross_entropy_with_logits(
                        presence_logits, batch_presence
                    )
                    loss = loss + presence_weight * presence_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        predicted, presence_logits, _, observed_indices = _evaluate_encoder(
            model, validation_loader, device=device
        )
        order = np.argsort(observed_indices)
        if burden_weight > 0.0:
            predicted_full = np.zeros_like(burdens)
            predicted_full[observed_indices[order]] = predicted[order]
            analytic = _analytic_prediction_metrics(
                predicted_full,
                validation_indices,
                validation_targets,
                relation_config=relation_config,
                scales=scales,
            )
            score = float(analytic.spectrum_macro_f1)
            endpoint = "ANALYTIC_MULTISCALE_MACRO_F1"
        else:
            truth_presence = presence[observed_indices[order]]
            logits_tensor = torch.from_numpy(presence_logits[order])
            truth_tensor = torch.from_numpy(truth_presence)
            validation_bce = float(
                functional.binary_cross_entropy_with_logits(logits_tensor, truth_tensor)
            )
            score = -validation_bce
            endpoint = "NEGATIVE_PRESENCE_BCE"
        print(
            json.dumps(
                {
                    "stage": "STAGE3A_ENCODER",
                    "contract": contract,
                    "seed": seed,
                    "epoch": epoch,
                    "early_stopping_endpoint": endpoint,
                    "score": score,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if score > best_score + float(settings["early_stopping_min_delta"]):
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(settings["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("Stage 3A encoder did not produce a finite stopping endpoint")
    model.load_state_dict(best_state)
    checkpoint_path = output_directory / f"encoder_{contract}_seed{seed}.pt"
    if checkpoint_path.exists():
        raise FileExistsError("Stage 3A encoder checkpoint already exists")
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "contract": contract,
            "seed": seed,
            "best_epoch": best_epoch,
            "early_stopping_score": best_score,
        },
        checkpoint_path,
    )
    all_loader = _loader(
        _Stage3AImageDataset(
            images,
            burdens,
            presence,
            np.arange(images.shape[0], dtype=np.int64),
            training=False,
            geometric_augmentation=False,
        ),
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        seed=seed,
        device=device,
    )
    predicted_all, _, embeddings, observed_indices = _evaluate_encoder(
        model, all_loader, device=device
    )
    resource = _resource_end(started, device)
    order = np.argsort(observed_indices)
    return EncoderArtifact(
        contract=contract,
        seed=seed,
        embeddings=embeddings[order],
        predicted_burdens=(predicted_all[order] if burden_weight > 0.0 else None),
        best_epoch=best_epoch,
        validation_score=best_score,
        resource=resource,
        parameter_count=parameter_count(model),
        trainable_parameter_count=parameter_count(model, trainable_only=True),
        checkpoint_sha256=sha256_file(checkpoint_path),
    )


def _pair_dataset(targets: PairTargets) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(targets.left_indices),
        torch.from_numpy(targets.right_indices),
        torch.from_numpy(targets.relation_codes),
        torch.from_numpy(targets.transition),
        torch.from_numpy(targets.critical_log2),
    )


def _four_state_weights(targets: PairTargets, maximum: float) -> Tensor:
    merged = targets.relation_codes.copy()
    merged[merged == 4] = 3
    counts = np.bincount(merged.ravel(), minlength=4).astype(np.float64)
    weights = np.zeros(4, dtype=np.float64)
    present = counts > 0
    weights[present] = 1.0 / np.sqrt(counts[present])
    weights[present] /= weights[present].mean()
    return torch.from_numpy(np.minimum(weights, maximum).astype(np.float32))


def _pair_model_predictions(
    head: ConfigurablePairHead | RankNetScalarHead,
    embeddings: Tensor,
    targets: PairTargets,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
    critical_mean: float,
    critical_std: float,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float32] | None,
    NDArray[np.float32] | None,
]:
    loader = _loader(
        _pair_dataset(targets),
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        device=device,
    )
    forward_rows: list[NDArray[np.int64]] = []
    reverse_rows: list[NDArray[np.int64]] = []
    transition_rows: list[NDArray[np.float32]] = []
    critical_rows: list[NDArray[np.float32]] = []
    head.eval()
    with torch.inference_mode():
        for left, right, _, _, _ in loader:
            left_embedding = embeddings[left.to(device)]
            right_embedding = embeddings[right.to(device)]
            if isinstance(head, RankNetScalarHead):
                forward_logits = head(left_embedding, right_embedding)
                reverse_logits = head(right_embedding, left_embedding)
                transition = None
                critical = None
            else:
                forward_logits, transition, critical = head(left_embedding, right_embedding)
                reverse_logits, _, _ = head(right_embedding, left_embedding)
            forward_rows.append(forward_logits.argmax(dim=2).cpu().numpy())
            reverse_rows.append(reverse_logits.argmax(dim=2).cpu().numpy())
            if transition is not None:
                transition_rows.append(torch.sigmoid(transition).float().cpu().numpy())
            if critical is not None:
                critical_rows.append(
                    (critical * critical_std + critical_mean).float().cpu().numpy()
                )
    return (
        np.concatenate(forward_rows),
        np.concatenate(reverse_rows),
        None if not transition_rows else np.concatenate(transition_rows),
        None if not critical_rows else np.concatenate(critical_rows),
    )


@dataclass(frozen=True, slots=True)
class PairArtifact:
    metrics: Stage3AMetrics
    prediction: NDArray[np.int64]
    reverse_prediction: NDArray[np.int64]
    transition_score: NDArray[np.float32] | None
    critical_prediction: NDArray[np.float32] | None
    best_epoch: int
    resource: ResourceUsage
    parameter_count: int
    checkpoint_sha256: str


def _train_pair_run(
    run_id: str,
    spec: Mapping[str, Any],
    embeddings_array: NDArray[np.float32],
    train_targets: PairTargets,
    validation_targets: PairTargets,
    *,
    output_directory: Path,
    seed: int,
    device: torch.device,
    common: Mapping[str, Any],
    scales: tuple[float, ...],
) -> PairArtifact:
    settings = common["pair_training"]
    losses = set(str(value) for value in spec.get("losses", ()))
    _seed_everything(seed)
    embeddings = torch.from_numpy(embeddings_array).to(device)
    is_ranknet = str(spec["head"]) == "ranknet_scalar_cutpoints"
    class_count = 4 if str(spec["head"]) == "merged_si_unresolved_four_state" else 5
    if is_ranknet:
        head: ConfigurablePairHead | RankNetScalarHead = RankNetScalarHead(
            scale_count=len(scales)
        ).to(device)
    else:
        representation_values = spec.get("pair_representation")
        if representation_values is None:
            representation: PairRepresentation = "full"
        else:
            value = tuple(str(item) for item in representation_values)
            mapping: dict[tuple[str, ...], PairRepresentation] = {
                ("signed_difference",): "signed_difference",
                ("absolute_difference",): "absolute_difference",
            }
            if value not in mapping:
                raise ValueError("Stage 3A pair representation contract is invalid")
            representation = mapping[value]
        head_kind = str(spec["head"])
        if head_kind not in {
            "joint_spectrum",
            "independent_scale",
            "lambda_conditioned_shared",
            "merged_si_unresolved_four_state",
        }:
            raise ValueError("Stage 3A pair head contract is invalid")
        head = ConfigurablePairHead(
            head_kind=head_kind,  # type: ignore[arg-type]
            pair_representation=representation,
            scale_count=len(scales),
            relation_count=class_count,
            dropout=float(settings["dropout"]),
            transition_output="transition" in losses,
            critical_output="critical" in losses,
            scales=scales,
        ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(settings["max_epochs"])
    )
    loader = _loader(
        _pair_dataset(train_targets),
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        seed=seed,
        device=device,
    )
    if str(spec.get("class_weighting", "DEFAULT")) == "NONE":
        class_weights = None
    elif class_count == 4:
        class_weights = _four_state_weights(
            train_targets, float(settings["maximum_class_weight"])
        ).to(device)
    else:
        class_weights = _class_weights(
            train_targets, maximum=float(settings["maximum_class_weight"])
        ).to(device)
    finite_critical = np.isfinite(train_targets.critical_log2)
    critical_mean = float(np.mean(train_targets.critical_log2[finite_critical]))
    critical_std = max(
        float(np.std(train_targets.critical_log2[finite_critical])), 1.0e-6
    )
    use_amp = bool(settings["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # type: ignore[attr-defined]
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    started = _resource_start(device)
    for epoch in range(1, int(settings["max_epochs"]) + 1):
        head.train()
        for left, right, relation, transition, critical in loader:
            left = left.to(device)
            right = right.to(device)
            relation = relation.to(device)
            transition = transition.to(device)
            critical = critical.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):  # type: ignore[attr-defined]
                if isinstance(head, RankNetScalarHead):
                    logits = head(embeddings[left], embeddings[right])
                    eligible = relation < 3
                    if not bool(eligible.any()):
                        continue
                    loss = functional.cross_entropy(logits[eligible], relation[eligible])
                else:
                    logits, transition_logit, critical_prediction = head(
                        embeddings[left], embeddings[right]
                    )
                    spectrum_truth = relation
                    if class_count == 4:
                        spectrum_truth = spectrum_truth.clone()
                        spectrum_truth[spectrum_truth == 4] = 3
                    loss = functional.cross_entropy(
                        logits.reshape(-1, class_count),
                        spectrum_truth.reshape(-1),
                        weight=class_weights,
                    )
                    if "transition" in losses:
                        assert transition_logit is not None
                        loss = loss + float(settings["transition_loss_weight"]) * (
                            functional.binary_cross_entropy_with_logits(
                                transition_logit, transition
                            )
                        )
                    if "critical" in losses:
                        assert critical_prediction is not None
                        selector = transition > 0.5
                        if bool(selector.any()):
                            standardized = (critical[selector] - critical_mean) / critical_std
                            critical_loss = functional.smooth_l1_loss(
                                critical_prediction[selector],
                                standardized,
                                beta=float(settings["critical_huber_beta"]),
                            )
                            loss = loss + float(settings["critical_loss_weight"]) * critical_loss
                    if "soft_admissibility" in losses:
                        loss = loss + float(settings["soft_admissibility_weight"]) * (
                            soft_admissibility_penalty(logits)
                        )
            scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        prediction, reverse, transition_score, critical_prediction_array = (
            _pair_model_predictions(
                head,
                embeddings,
                validation_targets,
                batch_size=int(settings["batch_size"]),
                seed=seed,
                device=device,
                critical_mean=critical_mean,
                critical_std=critical_std,
            )
        )
        metrics = evaluate_stage3a_predictions(
            validation_targets,
            prediction,
            reverse,
            transition_score=transition_score,
            critical_prediction=critical_prediction_array,
            class_count=class_count,
        )
        score = metrics.spectrum_macro_f1
        print(
            json.dumps(
                {
                    "stage": "STAGE3A_PAIR_HEAD",
                    "run_id": run_id,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_macro_f1": score,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if score > best_score + float(settings["early_stopping_min_delta"]):
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in head.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(settings["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("Stage 3A pair head did not produce a finite metric")
    head.load_state_dict(best_state)
    prediction, reverse, transition_score, critical_prediction_array = _pair_model_predictions(
        head,
        embeddings,
        validation_targets,
        batch_size=int(settings["batch_size"]),
        seed=seed,
        device=device,
        critical_mean=critical_mean,
        critical_std=critical_std,
    )
    metrics = evaluate_stage3a_predictions(
        validation_targets,
        prediction,
        reverse,
        transition_score=transition_score,
        critical_prediction=critical_prediction_array,
        class_count=class_count,
    )
    checkpoint_path = output_directory / f"head_{run_id}_seed{seed}.pt"
    if checkpoint_path.exists():
        raise FileExistsError("Stage 3A pair checkpoint already exists")
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "run_id": run_id,
            "seed": seed,
            "best_epoch": best_epoch,
            "critical_mean": critical_mean,
            "critical_std": critical_std,
        },
        checkpoint_path,
    )
    resource = _resource_end(started, device)
    return PairArtifact(
        metrics=metrics,
        prediction=prediction,
        reverse_prediction=reverse,
        transition_score=transition_score,
        critical_prediction=critical_prediction_array,
        best_epoch=best_epoch,
        resource=resource,
        parameter_count=parameter_count(head),
        checkpoint_sha256=sha256_file(checkpoint_path),
    )


def _aliases_for(run_id: str, runs: Mapping[str, Any]) -> list[str]:
    aliases = [str(alias) for alias, target in runs["aliases"].items() if target == run_id]
    return sorted({run_id, *aliases})


def _run_result_payload(
    *,
    run_id: str,
    aliases: Sequence[str],
    seed: int,
    spec: Mapping[str, Any],
    metrics: Stage3AMetrics,
    encoder: EncoderArtifact | None,
    pair: PairArtifact | None,
    code_commit: str,
    common_sha256: str,
    runs_sha256: str,
    cache_sha256: str,
    partition_digest: str,
    run_contract_sha256: str,
) -> dict[str, Any]:
    encoder_resource = ResourceUsage(0.0, 0) if encoder is None else encoder.resource
    pair_resource = ResourceUsage(0.0, 0) if pair is None else pair.resource
    encoder_parameters = 0 if encoder is None else encoder.parameter_count
    pair_parameters = 0 if pair is None else pair.parameter_count
    return {
        "status": "STAGE3A_DEVELOPMENT_RUN_COMPLETE",
        "evidence_level": "DEVELOPMENT_ONLY_SUPPORTIVE_EVIDENCE",
        "run_id": run_id,
        "aliases": list(aliases),
        "seed": seed,
        "tier": spec["tier"],
        "family": spec["family"],
        "metrics": metrics.to_dict(),
        "resource": {
            "encoder_runtime_seconds": encoder_resource.runtime_seconds,
            "head_runtime_seconds": pair_resource.runtime_seconds,
            "total_pipeline_runtime_seconds": (
                encoder_resource.runtime_seconds + pair_resource.runtime_seconds
            ),
            "peak_cuda_vram_bytes": max(
                encoder_resource.peak_cuda_vram_bytes,
                pair_resource.peak_cuda_vram_bytes,
            ),
            "encoder_parameter_count": encoder_parameters,
            "head_parameter_count": pair_parameters,
            "total_pipeline_parameter_count": encoder_parameters + pair_parameters,
            "best_encoder_epoch": None if encoder is None else encoder.best_epoch,
            "best_head_epoch": None if pair is None else pair.best_epoch,
        },
        "provenance": {
            "code_commit": code_commit,
            "common_config_sha256": common_sha256,
            "run_matrix_sha256": runs_sha256,
            "run_contract_sha256": run_contract_sha256,
            "development_cache_sha256": cache_sha256,
            "partition_digest": partition_digest,
            "encoder_checkpoint_sha256": (
                None if encoder is None else encoder.checkpoint_sha256
            ),
            "head_checkpoint_sha256": None if pair is None else pair.checkpoint_sha256,
        },
        "guards": {
            "official_test_accessed": False,
            "ddr_accessed": False,
            "formal_results_recomputed": False,
            "source_identifiers_recorded": False,
            "automatic_retry": False,
        },
    }


def _save_run(
    output_root: Path,
    payload: Mapping[str, Any],
    truth: PairTargets,
    prediction: NDArray[np.int64],
    reverse_prediction: NDArray[np.int64],
    transition_score: NDArray[np.float32] | None,
    critical_prediction: NDArray[np.float32] | None,
) -> None:
    run_id = str(payload["run_id"])
    seed = int(payload["seed"])
    directory = output_root / "runs" / run_id / f"seed_{seed}"
    if directory.exists():
        raise FileExistsError("Stage 3A run directory already exists")
    _write_json_once(directory / "result.json", payload)
    arrays: dict[str, NDArray[Any]] = {
        "truth": truth.relation_codes,
        "prediction": prediction,
        "reverse_prediction": reverse_prediction,
        "left_node": truth.left_indices,
        "right_node": truth.right_indices,
    }
    if transition_score is not None:
        arrays["transition_score"] = transition_score
    if critical_prediction is not None:
        arrays["critical_prediction"] = critical_prediction
    _write_npz_once(directory / "validation_predictions.npz", **arrays)


def run_stage3a_development(
    repository_root: str | Path,
    cache_path: str | Path,
    pretrained_weights: str | Path,
    common_config_path: str | Path,
    runs_config_path: str | Path,
    output_root: str | Path,
    *,
    include_p1: bool,
) -> dict[str, Any]:
    """Execute the frozen Stage 3A matrix on development data only."""

    root = Path(repository_root).resolve()
    cache_file = Path(cache_path).resolve()
    weights_file = Path(pretrained_weights).resolve()
    common_file = Path(common_config_path).resolve()
    runs_file = Path(runs_config_path).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError("Stage 3A output root already exists")
    expected_parent = (root / "reports" / "stage3a").resolve()
    if expected_parent not in output.parents:
        raise ValueError("Stage 3A output must remain inside reports/stage3a")
    common: Mapping[str, Any] = yaml.safe_load(common_file.read_text(encoding="utf-8"))
    runs: Mapping[str, Any] = yaml.safe_load(runs_file.read_text(encoding="utf-8"))
    forbidden = tuple(
        str(value).lower()
        for value in common["denylist"]["forbidden_argument_tokens"]
    )
    for candidate in (str(cache_file), str(weights_file), str(output)):
        lowered = candidate.lower()
        if any(token in lowered for token in forbidden):
            raise ValueError("Stage 3A denylist rejected an input path")
    if cache_file.name != str(common["data"]["allowed_cache_basename"]):
        raise ValueError("Stage 3A cache basename is not frozen")
    cache_hash = sha256_file(cache_file)
    if cache_hash != str(common["data"]["allowed_cache_sha256"]):
        raise ValueError("Stage 3A development cache hash mismatch")
    if sha256_file(weights_file) != str(common["pretraining"]["sha256"]):
        raise ValueError("Stage 3A ImageNet weight hash mismatch")
    _require_tracked_clean(root)
    code_commit = _git_commit(root)
    cache = load_tjdr_model_cache(cache_file)
    images = np.asarray(cache["images"], dtype=np.uint8)
    burdens = np.asarray(cache["burdens"], dtype=np.float32)
    presence = np.asarray(cache["presence"], dtype=np.float32)
    partition = np.asarray(cache["partition"], dtype=np.uint8)
    train_indices = np.flatnonzero(partition == 0).astype(np.int64)
    validation_indices = np.flatnonzero(partition == 1).astype(np.int64)
    expected = common["data"]
    if (
        images.shape[0] != int(expected["expected_unique_content_units"])
        or train_indices.size != int(expected["expected_train_units"])
        or validation_indices.size != int(expected["expected_validation_units"])
    ):
        raise ValueError("Stage 3A development unit boundary mismatch")
    relation_config = _relation_config(common)
    scales = _scales(common)
    train_targets = build_relation_targets(
        burdens, train_indices.tolist(), base_config=relation_config, scales=scales
    )
    validation_targets = build_relation_targets(
        burdens, validation_indices.tolist(), base_config=relation_config, scales=scales
    )
    if (
        train_targets.pair_count != int(expected["expected_train_pairs"])
        or validation_targets.pair_count != int(expected["expected_validation_pairs"])
    ):
        raise ValueError("Stage 3A within-partition pair boundary mismatch")
    output.mkdir(parents=True, exist_ok=False)
    common_hash = sha256_file(common_file)
    runs_hash = sha256_file(runs_file)
    _write_json_once(
        output / "EXECUTION_MANIFEST.json",
        {
            "status": "STAGE3A_DEVELOPMENT_EXECUTION_STARTED",
            "evidence_level": "DEVELOPMENT_ONLY_SUPPORTIVE_EVIDENCE",
            "code_commit": code_commit,
            "common_config_sha256": common_hash,
            "run_matrix_sha256": runs_hash,
            "development_cache_sha256": cache_hash,
            "partition_digest": expected["partition_digest"],
            "include_p1": include_p1,
            "official_test_accessed": False,
            "ddr_accessed": False,
            "formal_results_recomputed": False,
        },
    )
    device = _device_from_config(common)
    seeds = tuple(int(value) for value in common["seeds"])
    encoder_specs: Mapping[str, Mapping[str, Any]] = runs["encoder_contracts"]
    required_contracts = {"frozen_imagenet", "burden_only", "presence_only", "burden_presence"}
    if include_p1:
        required_contracts.add("burden_presence_no_geometry")
    completed_runs = 0
    for seed in seeds:
        encoders: dict[str, EncoderArtifact] = {}
        encoders["frozen_imagenet"] = _frozen_encoder_embeddings(
            images,
            burdens,
            presence,
            weights_path=weights_file,
            batch_size=int(common["encoder_training"]["batch_size"]),
            seed=seed,
            device=device,
        )
        for contract in sorted(required_contracts - {"frozen_imagenet"}):
            encoders[contract] = _train_encoder_contract(
                contract,
                encoder_specs[contract],
                images,
                burdens,
                presence,
                train_indices,
                validation_indices,
                validation_targets,
                relation_config=relation_config,
                scales=scales,
                weights_path=weights_file,
                output_directory=output / "encoders" / contract / f"seed_{seed}",
                seed=seed,
                device=device,
                common=common,
            )
        for run_id_value in runs["execution_order"]:
            run_id = str(run_id_value)
            spec: Mapping[str, Any] = runs["unique_runs"][run_id]
            if str(spec["tier"]) == "P1" and not include_p1:
                continue
            run_contract_hash = _canonical_hash(
                {
                    "common_config_sha256": common_hash,
                    "run_matrix_sha256": runs_hash,
                    "run_id": run_id,
                    "spec": spec,
                    "seed": seed,
                }
            )
            aliases = _aliases_for(run_id, runs)
            family = str(spec["family"])
            encoder: EncoderArtifact | None = None
            pair: PairArtifact | None = None
            if family == "majority":
                prediction = majority_spectrum_baseline(
                    train_targets, validation_targets.pair_count
                )
                reverse = prediction.copy()
                metrics = evaluate_stage3a_predictions(
                    validation_targets, prediction, reverse
                )
                transition_score = None
                critical_prediction = None
            elif family == "analytic":
                encoder = encoders[str(spec["encoder"])]
                if encoder.predicted_burdens is None:
                    raise RuntimeError("analytic Stage 3A run lacks burden predictions")
                predicted_targets = build_relation_targets(
                    encoder.predicted_burdens,
                    validation_indices.tolist(),
                    base_config=relation_config,
                    scales=scales,
                )
                prediction = predicted_targets.relation_codes
                reverse = _INVERSION[prediction]
                transition_score = predicted_targets.transition
                critical_prediction = predicted_targets.critical_log2.copy()
                missing = np.logical_and(
                    np.isfinite(validation_targets.critical_log2),
                    ~np.isfinite(critical_prediction),
                )
                critical_prediction[missing] = 0.0
                metrics = evaluate_stage3a_predictions(
                    validation_targets,
                    prediction,
                    reverse,
                    transition_score=transition_score,
                    critical_prediction=critical_prediction,
                )
            else:
                encoder = encoders[str(spec["encoder"])]
                pair = _train_pair_run(
                    run_id,
                    spec,
                    encoder.embeddings,
                    train_targets,
                    validation_targets,
                    output_directory=output / "heads" / run_id / f"seed_{seed}",
                    seed=seed,
                    device=device,
                    common=common,
                    scales=scales,
                )
                prediction = pair.prediction
                reverse = pair.reverse_prediction
                transition_score = pair.transition_score
                critical_prediction = pair.critical_prediction
                metrics = pair.metrics
            payload = _run_result_payload(
                run_id=run_id,
                aliases=aliases,
                seed=seed,
                spec=spec,
                metrics=metrics,
                encoder=encoder,
                pair=pair,
                code_commit=code_commit,
                common_sha256=common_hash,
                runs_sha256=runs_hash,
                cache_sha256=cache_hash,
                partition_digest=str(expected["partition_digest"]),
                run_contract_sha256=run_contract_hash,
            )
            _save_run(
                output,
                payload,
                validation_targets,
                prediction,
                reverse,
                transition_score,
                critical_prediction,
            )
            completed_runs += 1
            print(
                json.dumps(
                    {
                        "stage": "STAGE3A_RUN_COMPLETE",
                        "run_id": run_id,
                        "seed": seed,
                        "macro_f1": metrics.spectrum_macro_f1,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del encoders
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = {
        "status": "STAGE3A_DEVELOPMENT_EXECUTION_COMPLETE",
        "evidence_level": "DEVELOPMENT_ONLY_SUPPORTIVE_EVIDENCE",
        "completed_unique_seed_runs": completed_runs,
        "seed_count": len(seeds),
        "include_p1": include_p1,
        "code_commit": code_commit,
        "common_config_sha256": common_hash,
        "run_matrix_sha256": runs_hash,
        "development_cache_sha256": cache_hash,
        "partition_digest": expected["partition_digest"],
        "official_test_accessed": False,
        "ddr_accessed": False,
        "formal_results_recomputed": False,
        "automatic_retry": False,
    }
    _write_json_once(output / "EXECUTION_COMPLETE.json", summary)
    return summary


__all__ = ["run_stage3a_development"]

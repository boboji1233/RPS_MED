"""Append-only development execution for the RPS-Med Stage 3B path grammar."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as functional
import yaml
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from nts_med.reproducibility.freeze import sha256_file
from rps_med.model_data import load_tjdr_model_cache
from rps_med.model_metrics import PairTargets, build_relation_targets
from rps_med.model_training import _class_weights, _relation_config, _scales, _seed_everything
from rps_med.modeling import BurdenEncoder
from rps_med.stage3a_models import (
    ConfigurablePairHead,
    Stage3AMetrics,
    evaluate_stage3a_predictions,
    parameter_count,
)
from rps_med.stage3b_grammar import (
    canonical_path_ids,
    grammar_decode,
    minimum_hamming_projection,
    structured_path_nll,
)

_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError("Stage 3B append-only JSON already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_npz_once(path: Path, **arrays: NDArray[Any]) -> None:
    if path.exists():
        raise FileExistsError("Stage 3B append-only prediction artifact already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Stage 3B configuration must be a mapping")
    return payload


def _require_isolated_environment(config: Mapping[str, Any]) -> None:
    expected = Path(str(config["runtime"]["environment"])).resolve()
    actual = Path(sys.executable).resolve().parent
    if os.path.normcase(str(expected)) != os.path.normcase(str(actual)):
        raise RuntimeError("Stage 3B must run in its frozen isolated environment")


def _require_safe_paths(paths: Sequence[Path], config: Mapping[str, Any]) -> None:
    forbidden = tuple(str(value).lower() for value in config["denylist"]["forbidden_tokens"])
    for path in paths:
        lowered = str(path).lower()
        if any(token in lowered for token in forbidden):
            raise ValueError("Stage 3B received a forbidden input path")


def _input_paths(stage3a_root: Path, seed: int) -> tuple[Path, Path, Path]:
    encoder = (
        stage3a_root
        / "encoders"
        / "burden_presence"
        / f"seed_{seed}"
        / f"encoder_burden_presence_seed{seed}.pt"
    )
    head = (
        stage3a_root
        / "heads"
        / "B6_A0_A12full"
        / f"seed_{seed}"
        / f"head_B6_A0_A12full_seed{seed}.pt"
    )
    predictions = (
        stage3a_root
        / "runs"
        / "B6_A0_A12full"
        / f"seed_{seed}"
        / "validation_predictions.npz"
    )
    return encoder, head, predictions


def _validate_fixed_inputs(
    config: Mapping[str, Any], stage3a_root: Path, cache_path: Path
) -> dict[int, tuple[Path, Path, Path]]:
    expected_cache = str(config["data"]["allowed_cache_sha256"])
    if sha256_file(cache_path) != expected_cache:
        raise ValueError("Stage 3B development cache binding failed")
    output: dict[int, tuple[Path, Path, Path]] = {}
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        encoder, head, predictions = _input_paths(stage3a_root, seed)
        expected = config["fixed_inputs"][str(seed)]
        observed = (
            sha256_file(encoder),
            sha256_file(head),
            sha256_file(predictions),
        )
        required = (
            str(expected["encoder_sha256"]),
            str(expected["full_head_sha256"]),
            str(expected["full_predictions_sha256"]),
        )
        if observed != required:
            raise ValueError("Stage 3B fixed Stage 3A input binding failed")
        output[seed] = (encoder, head, predictions)
    return output


def _configure_device(config: Mapping[str, Any]) -> torch.device:
    if str(config["runtime"]["device"]) != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage 3B frozen protocol requires CUDA")
    device = torch.device("cuda:0")
    fraction = float(config["runtime"]["torch_memory_fraction"])
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < int(config["runtime"]["max_process_cuda_bytes"]):
        raise RuntimeError("Stage 3B launch has insufficient free CUDA memory")
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
    if device.type != "cuda":
        return ResourceUsage(runtime_seconds=time.perf_counter() - start, peak_cuda_vram_bytes=0)
    torch.cuda.synchronize(device)
    return ResourceUsage(
        runtime_seconds=time.perf_counter() - start,
        peak_cuda_vram_bytes=int(torch.cuda.max_memory_allocated(device)),
    )


def _extract_fixed_embeddings(
    images: NDArray[np.uint8], checkpoint_path: Path, *, device: torch.device
) -> tuple[NDArray[np.float32], ResourceUsage]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Stage 3B encoder checkpoint schema is invalid")
    model = BurdenEncoder()
    model.load_state_dict(checkpoint["model"])
    for value in model.parameters():
        value.requires_grad = False
    model.to(device).eval()
    mean = _MEAN.to(device)
    std = _STD.to(device)
    rows: list[NDArray[np.float32]] = []
    started = _resource_start(device)
    with torch.inference_mode():
        for first in range(0, images.shape[0], 8):
            array = torch.from_numpy(images[first : first + 8].copy())
            batch = array.permute(0, 3, 1, 2).float().to(device) / 255.0
            embedding, _, _ = model((batch - mean) / std)
            rows.append(embedding.float().cpu().numpy())
    resource = _resource_end(started, device)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(rows).astype(np.float32, copy=False), resource


def _pair_loader(
    targets: PairTargets,
    *,
    path_ids: NDArray[np.int64] | None,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[Any]:
    arrays: list[Tensor] = [
        torch.from_numpy(targets.left_indices),
        torch.from_numpy(targets.right_indices),
        torch.from_numpy(targets.relation_codes),
    ]
    if path_ids is not None:
        arrays.append(torch.from_numpy(path_ids))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(*arrays),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )


def _replay_logits(
    head: ConfigurablePairHead,
    embeddings: Tensor,
    targets: PairTargets,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    forward_rows: list[Tensor] = []
    reverse_rows: list[Tensor] = []
    head.eval()
    with torch.inference_mode():
        for left, right, _ in _pair_loader(
            targets,
            path_ids=None,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
        ):
            left_embedding = embeddings[left.to(device)]
            right_embedding = embeddings[right.to(device)]
            forward, _, _ = head(left_embedding, right_embedding)
            reverse, _, _ = head(right_embedding, left_embedding)
            forward_rows.append(forward)
            reverse_rows.append(reverse)
    return torch.cat(forward_rows), torch.cat(reverse_rows)


def _prediction_metrics(
    targets: PairTargets,
    prediction: NDArray[np.int64],
    reverse: NDArray[np.int64],
) -> Stage3AMetrics:
    return evaluate_stage3a_predictions(targets, prediction, reverse)


def _load_reference_predictions(
    path: Path, targets: PairTargets
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"truth", "prediction", "reverse_prediction", "left_node", "right_node"}
        if not required.issubset(archive.files):
            raise ValueError("Stage 3B reference prediction schema is invalid")
        truth = np.asarray(archive["truth"], dtype=np.int64)
        prediction = np.asarray(archive["prediction"], dtype=np.int64)
        reverse = np.asarray(archive["reverse_prediction"], dtype=np.int64)
        left = np.asarray(archive["left_node"], dtype=np.int64)
        right = np.asarray(archive["right_node"], dtype=np.int64)
    if not (
        np.array_equal(truth, targets.relation_codes)
        and np.array_equal(left, targets.left_indices)
        and np.array_equal(right, targets.right_indices)
    ):
        raise ValueError("Stage 3B reference predictions do not match frozen targets")
    return prediction, reverse


def _load_reference_head(
    path: Path, *, scales: tuple[float, ...], device: torch.device
) -> ConfigurablePairHead:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Stage 3B reference head checkpoint schema is invalid")
    head = ConfigurablePairHead(
        head_kind="joint_spectrum",
        pair_representation="full",
        scale_count=len(scales),
        relation_count=5,
        dropout=0.20,
        transition_output=True,
        critical_output=True,
        scales=scales,
    )
    head.load_state_dict(checkpoint["model"])
    return head.to(device)


@dataclass(frozen=True, slots=True)
class TrainedHead:
    metrics: Stage3AMetrics
    prediction: NDArray[np.int64]
    reverse_prediction: NDArray[np.int64]
    best_epoch: int
    parameter_count: int
    resource: ResourceUsage
    checkpoint_sha256: str


def _grammar_predictions(
    head: ConfigurablePairHead,
    embeddings: Tensor,
    targets: PairTargets,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    forward, reverse = _replay_logits(
        head, embeddings, targets, batch_size=batch_size, device=device
    )
    return (
        grammar_decode(forward).cpu().numpy().astype(np.int64, copy=False),
        grammar_decode(reverse).cpu().numpy().astype(np.int64, copy=False),
    )


def _train_grammar_head(
    method_id: str,
    embeddings_array: NDArray[np.float32],
    train_targets: PairTargets,
    validation_targets: PairTargets,
    *,
    seed: int,
    scales: tuple[float, ...],
    config: Mapping[str, Any],
    output_directory: Path,
    device: torch.device,
) -> TrainedHead:
    settings = config["training"]
    method = config["methods"][method_id]
    _seed_everything(seed)
    embeddings = torch.from_numpy(embeddings_array).to(device)
    head = ConfigurablePairHead(
        head_kind="joint_spectrum",
        pair_representation="full",
        scale_count=len(scales),
        relation_count=5,
        dropout=float(settings["dropout"]),
        transition_output=False,
        critical_output=False,
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
    train_path_ids = canonical_path_ids(train_targets.relation_codes)
    loader = _pair_loader(
        train_targets,
        path_ids=train_path_ids,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        seed=seed,
    )
    class_weights = _class_weights(
        train_targets, maximum=float(settings["maximum_class_weight"])
    ).to(device)
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    started = _resource_start(device)
    for epoch in range(1, int(settings["max_epochs"]) + 1):
        head.train()
        for left, right, relation, path_identifier in loader:
            optimizer.zero_grad(set_to_none=True)
            left = left.to(device)
            right = right.to(device)
            relation = relation.to(device)
            path_identifier = path_identifier.to(device)
            logits, _, _ = head(embeddings[left], embeddings[right])
            loss = float(method["structured_weight"]) * structured_path_nll(
                logits, path_identifier
            )
            if float(method["scale_ce_weight"]) > 0.0:
                loss = loss + float(method["scale_ce_weight"]) * functional.cross_entropy(
                    logits.reshape(-1, 5), relation.reshape(-1), weight=class_weights
                )
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        scheduler.step()
        prediction, reverse = _grammar_predictions(
            head,
            embeddings,
            validation_targets,
            batch_size=int(settings["batch_size"]),
            device=device,
        )
        metrics = _prediction_metrics(validation_targets, prediction, reverse)
        score = metrics.spectrum_macro_f1
        print(
            json.dumps(
                {
                    "stage": "STAGE3B_GRAMMAR_HEAD",
                    "method_id": method_id,
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
        raise RuntimeError("Stage 3B grammar head did not produce a finite metric")
    head.load_state_dict(best_state)
    prediction, reverse = _grammar_predictions(
        head,
        embeddings,
        validation_targets,
        batch_size=int(settings["batch_size"]),
        device=device,
    )
    metrics = _prediction_metrics(validation_targets, prediction, reverse)
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / f"head_{method_id}_seed{seed}.pt"
    if checkpoint_path.exists():
        raise FileExistsError("Stage 3B grammar checkpoint already exists")
    torch.save(
        {"model": best_state, "method_id": method_id, "seed": seed, "best_epoch": best_epoch},
        checkpoint_path,
    )
    resource = _resource_end(started, device)
    return TrainedHead(
        metrics=metrics,
        prediction=prediction,
        reverse_prediction=reverse,
        best_epoch=best_epoch,
        parameter_count=parameter_count(head),
        resource=resource,
        checkpoint_sha256=sha256_file(checkpoint_path),
    )


def _write_result(
    output_root: Path,
    method_id: str,
    seed: int,
    targets: PairTargets,
    prediction: NDArray[np.int64],
    reverse: NDArray[np.int64],
    metrics: Stage3AMetrics,
    *,
    resource: ResourceUsage,
    best_epoch: int | None,
    trainable_parameters: int,
    provenance: Mapping[str, Any],
) -> None:
    directory = output_root / "runs" / method_id / f"seed_{seed}"
    _write_npz_once(
        directory / "validation_predictions.npz",
        truth=targets.relation_codes,
        prediction=prediction,
        reverse_prediction=reverse,
        left_node=targets.left_indices,
        right_node=targets.right_indices,
    )
    _write_json_once(
        directory / "result.json",
        {
            "status": "STAGE3B_DEVELOPMENT_RUN_COMPLETE",
            "evidence_level": "DEVELOPMENT_ONLY_SUPPORTIVE_EVIDENCE",
            "method_id": method_id,
            "seed": seed,
            "metrics": metrics.to_dict(),
            "resource": {
                **asdict(resource),
                "best_epoch": best_epoch,
                "trainable_parameter_count": trainable_parameters,
            },
            "provenance": dict(provenance),
            "guards": {
                "official_test_accessed": False,
                "ddr_accessed": False,
                "formal_results_recomputed": False,
                "encoder_updated": False,
                "source_identifiers_recorded": False,
                "automatic_retry": False,
            },
        },
    )


def run_stage3b_grammar(
    repository_root: str | Path,
    config_path: str | Path,
    cache_path: str | Path,
    stage3a_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Execute the frozen Stage 3B development protocol once, append-only."""

    repository = Path(repository_root).resolve()
    config_file = Path(config_path).resolve()
    cache_file = Path(cache_path).resolve()
    stage3a_directory = Path(stage3a_root).resolve()
    output_directory = Path(output_root).resolve()
    if not repository.is_dir():
        raise ValueError("Stage 3B repository root is invalid")
    if output_directory.exists():
        raise FileExistsError("Stage 3B append-only output root already exists")
    config = _load_yaml(config_file)
    _require_isolated_environment(config)
    _require_safe_paths((config_file, cache_file, stage3a_directory, output_directory), config)
    fixed_inputs = _validate_fixed_inputs(config, stage3a_directory, cache_file)
    device = _configure_device(config)
    cache = load_tjdr_model_cache(cache_file)
    images = np.asarray(cache["images"], dtype=np.uint8)
    burdens = np.asarray(cache["burdens"], dtype=np.float32)
    partition = np.asarray(cache["partition"], dtype=np.uint8)
    expected = config["data"]
    train_indices = np.flatnonzero(partition == 0).astype(np.int64)
    validation_indices = np.flatnonzero(partition == 1).astype(np.int64)
    if (
        images.shape[0] != int(expected["unique_units"])
        or train_indices.size != int(expected["train_units"])
        or validation_indices.size != int(expected["validation_units"])
    ):
        raise ValueError("Stage 3B development partition contract failed")
    relation_config = _relation_config(config)
    scales = _scales(config)
    train_targets = build_relation_targets(
        burdens, train_indices.tolist(), base_config=relation_config, scales=scales
    )
    validation_targets = build_relation_targets(
        burdens, validation_indices.tolist(), base_config=relation_config, scales=scales
    )
    if (
        train_targets.pair_count != int(expected["train_pairs"])
        or validation_targets.pair_count != int(expected["validation_pairs"])
    ):
        raise ValueError("Stage 3B pair-count contract failed")
    canonical_path_ids(train_targets.relation_codes)
    canonical_path_ids(validation_targets.relation_codes)
    output_directory.mkdir(parents=True, exist_ok=False)
    _write_json_once(
        output_directory / "EXECUTION_MANIFEST.json",
        {
            "status": "STAGE3B_DEVELOPMENT_EXECUTION_STARTED",
            "config_sha256": sha256_file(config_file),
            "cache_sha256": sha256_file(cache_file),
            "methods": list(config["methods"]),
            "seeds": [int(value) for value in config["seeds"]],
            "official_test_accessed": False,
            "ddr_accessed": False,
        },
    )
    completed: list[dict[str, Any]] = []
    max_allowed = int(config["runtime"]["max_process_cuda_bytes"])
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        encoder_path, head_path, prediction_path = fixed_inputs[seed]
        reference_prediction, reference_reverse = _load_reference_predictions(
            prediction_path, validation_targets
        )
        zero_resource = ResourceUsage(runtime_seconds=0.0, peak_cuda_vram_bytes=0)
        reference_metrics = _prediction_metrics(
            validation_targets, reference_prediction, reference_reverse
        )
        common_provenance = {
            "config_sha256": sha256_file(config_file),
            "cache_sha256": sha256_file(cache_file),
            "encoder_sha256": sha256_file(encoder_path),
            "reference_head_sha256": sha256_file(head_path),
            "reference_predictions_sha256": sha256_file(prediction_path),
        }
        _write_result(
            output_directory,
            "G0",
            seed,
            validation_targets,
            reference_prediction,
            reference_reverse,
            reference_metrics,
            resource=zero_resource,
            best_epoch=None,
            trainable_parameters=0,
            provenance=common_provenance,
        )
        hard_prediction = minimum_hamming_projection(reference_prediction)
        hard_reverse = minimum_hamming_projection(reference_reverse)
        hard_metrics = _prediction_metrics(validation_targets, hard_prediction, hard_reverse)
        _write_result(
            output_directory,
            "G1",
            seed,
            validation_targets,
            hard_prediction,
            hard_reverse,
            hard_metrics,
            resource=zero_resource,
            best_epoch=None,
            trainable_parameters=0,
            provenance=common_provenance,
        )
        embeddings_array, embedding_resource = _extract_fixed_embeddings(
            images, encoder_path, device=device
        )
        if "G2" in config["methods"]:
            embeddings = torch.from_numpy(embeddings_array).to(device)
            replay_head = _load_reference_head(head_path, scales=scales, device=device)
            replay_started = _resource_start(device)
            replay_logits, replay_reverse_logits = _replay_logits(
                replay_head,
                embeddings,
                validation_targets,
                batch_size=int(config["training"]["batch_size"]),
                device=device,
            )
            if not (
                np.array_equal(
                    replay_logits.argmax(dim=2).cpu().numpy(), reference_prediction
                )
                and np.array_equal(
                    replay_reverse_logits.argmax(dim=2).cpu().numpy(), reference_reverse
                )
            ):
                raise ValueError(
                    "Stage 3B fixed-head replay does not match Stage 3A predictions"
                )
            likelihood_prediction = grammar_decode(replay_logits).cpu().numpy()
            likelihood_reverse = grammar_decode(replay_reverse_logits).cpu().numpy()
            replay_resource = _resource_end(replay_started, device)
            likelihood_metrics = _prediction_metrics(
                validation_targets, likelihood_prediction, likelihood_reverse
            )
            _write_result(
                output_directory,
                "G2",
                seed,
                validation_targets,
                likelihood_prediction,
                likelihood_reverse,
                likelihood_metrics,
                resource=replay_resource,
                best_epoch=None,
                trainable_parameters=0,
                provenance=common_provenance,
            )
            del replay_head, replay_logits, replay_reverse_logits, embeddings
            torch.cuda.empty_cache()
        for method_id in ("G3", "G4"):
            if method_id not in config["methods"]:
                continue
            trained = _train_grammar_head(
                method_id,
                embeddings_array,
                train_targets,
                validation_targets,
                seed=seed,
                scales=scales,
                config=config,
                output_directory=output_directory
                / "heads"
                / method_id
                / f"seed_{seed}",
                device=device,
            )
            if trained.resource.peak_cuda_vram_bytes >= max_allowed:
                raise RuntimeError("Stage 3B process exceeded the frozen CUDA ceiling")
            _write_result(
                output_directory,
                method_id,
                seed,
                validation_targets,
                trained.prediction,
                trained.reverse_prediction,
                trained.metrics,
                resource=trained.resource,
                best_epoch=trained.best_epoch,
                trainable_parameters=trained.parameter_count,
                provenance={
                    **common_provenance,
                    "checkpoint_sha256": trained.checkpoint_sha256,
                },
            )
        if embedding_resource.peak_cuda_vram_bytes >= max_allowed:
            raise RuntimeError("Stage 3B embedding replay exceeded the CUDA ceiling")
        completed.append(
            {
                "seed": seed,
                "embedding_runtime_seconds": embedding_resource.runtime_seconds,
                "embedding_peak_cuda_vram_bytes": embedding_resource.peak_cuda_vram_bytes,
                "methods": list(config["methods"]),
            }
        )
    method_count = len(config["methods"])
    trained_method_count = sum(
        bool(spec.get("training")) for spec in config["methods"].values()
    )
    summary = {
        "status": "STAGE3B_DEVELOPMENT_EXECUTION_COMPLETE",
        "repository_root_recorded": False,
        "config_sha256": sha256_file(config_file),
        "cache_sha256": sha256_file(cache_file),
        "completed": completed,
        "run_seed_units": len(completed) * method_count,
        "trained_run_seed_units": len(completed) * trained_method_count,
        "official_test_accessed": False,
        "ddr_accessed": False,
        "formal_results_recomputed": False,
        "automatic_retry": False,
        "remote_push": False,
    }
    _write_json_once(output_directory / "EXECUTION_COMPLETE.json", summary)
    return summary


__all__ = ["run_stage3b_grammar"]

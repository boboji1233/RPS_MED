"""Rank local cross-split duplicate candidates with locked frozen features."""

from __future__ import annotations

import argparse
import csv
import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from nts_med.cli._common import print_status, software_versions, write_json
from nts_med.data.adapters.base import AdapterStatus, DatasetSample
from nts_med.data.adapters.registry import create_adapter
from nts_med.data.config import DatasetConfig, DatasetConfigError, load_dataset_config
from nts_med.data.feature_similarity import find_cross_split_feature_candidates
from nts_med.data.manifests import sha256_file


def build_parser() -> argparse.ArgumentParser:
    """Return the local-only frozen-feature audit parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Rank cross-split duplicate candidates with a SHA-256-locked ResNet-50; "
            "never downloads weights, deletes images, or changes splits."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--metadata-root", type=Path, default=None)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-similarity", type=float, default=0.995)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def _configured_root(config: DatasetConfig, explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        return explicit_root
    import os

    configured = os.environ.get(config.root_env_var)
    return Path(configured) if configured else Path("data") / "raw" / config.name


def _configured_optional_root(
    environment_name: str | None,
    explicit_root: Path | None,
) -> Path | None:
    if explicit_root is not None:
        return explicit_root
    if environment_name is None:
        return None
    import os

    configured = os.environ.get(environment_name)
    return Path(configured) if configured else None


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_output_location(
    output: Path,
    roots: Sequence[Path | None],
) -> None:
    destination = output.resolve(strict=False)
    for root in roots:
        if root is not None and _is_within(destination, root.resolve(strict=False)):
            raise ValueError("feature-audit output must remain outside every dataset root")


def _validated_digest(value: str) -> str:
    normalized = value.casefold()
    if len(normalized) != 64:
        raise ValueError("expected weights SHA-256 must contain 64 hexadecimal digits")
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise ValueError("expected weights SHA-256 must be hexadecimal") from exc
    return normalized


def _load_runtime() -> tuple[Any, Any]:
    try:
        torch = importlib.import_module("torch")
        models = importlib.import_module("torchvision.models")
    except ImportError as exc:
        raise RuntimeError("torch and torchvision are required for this optional audit") from exc
    return torch, models


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not bool(torch.cuda.is_available()):
            raise ValueError("CUDA was requested but is unavailable")
        return "cuda"
    return "cuda" if bool(torch.cuda.is_available()) else "cpu"


def _load_model(weights_path: Path, device: str) -> tuple[Any, Any, int]:
    torch, models = _load_runtime()
    model = models.resnet50(weights=None)
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    transform = models.ResNet50_Weights.DEFAULT.transforms()
    return model, transform, 2048


def _extract_features(
    samples: Sequence[DatasetSample],
    *,
    model: Any,
    transform: Any,
    device: str,
    batch_size: int,
) -> tuple[dict[str, np.ndarray[Any, np.dtype[np.floating[Any]]]], dict[str, str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    torch, _ = _load_runtime()
    features: dict[str, np.ndarray[Any, np.dtype[np.floating[Any]]]] = {}
    split_by_instance: dict[str, str] = {}
    ordered = tuple(sorted(samples, key=lambda sample: sample.sample_id))
    with torch.inference_mode():
        for start in range(0, len(ordered), batch_size):
            batch_samples = ordered[start : start + batch_size]
            tensors = []
            for sample in batch_samples:
                with Image.open(sample.image_path) as image:
                    canonical = ImageOps.exif_transpose(image).convert("RGB")
                    tensors.append(transform(canonical))
            batch = torch.stack(tensors).to(device)
            embeddings = model(batch).detach().to("cpu").numpy()
            for sample, vector in zip(batch_samples, embeddings, strict=True):
                features[sample.sample_id] = np.asarray(vector, dtype=np.float64)
                split = sample.official_split
                split_by_instance[sample.sample_id] = (
                    split if isinstance(split, str) and split else "UNKNOWN"
                )
    return features, split_by_instance


def _write_candidates(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "left_id",
        "right_id",
        "left_split",
        "right_split",
        "cosine_similarity",
        "selected_by_threshold",
        "selected_by_top_k",
        "review_decision",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one local, locked-weight feature-similarity audit."""

    args = build_parser().parse_args(argv)
    try:
        config = load_dataset_config(args.config)
        root = _configured_root(config, args.root)
        image_root = _configured_optional_root(config.image_root_env_var, args.image_root)
        metadata_root = _configured_optional_root(
            config.additional_metadata_root_env_var,
            args.metadata_root,
        )
        _validate_output_location(args.output, (root, image_root, metadata_root))
        expected_digest = _validated_digest(args.expected_weights_sha256)
        observed_digest = sha256_file(args.weights)
        if observed_digest != expected_digest:
            raise ValueError("weights SHA-256 does not match the locked expected value")
        adapter = create_adapter(
            config.name,
            root,
            image_root=image_root,
            metadata_root=metadata_root,
            release_profile=config.additional_metadata_profile,
            component_acquisition_records=config.component_acquisition_records,
        )
        validation = adapter.validate()
        if validation.status is not AdapterStatus.READY:
            raise ValueError("dataset adapter must be READY before feature audit")
        samples = tuple(adapter.samples())
        torch, _ = _load_runtime()
        device = _resolve_device(torch, args.device)
        model, transform, feature_dimension = _load_model(args.weights, device)
        features, split_by_instance = _extract_features(
            samples,
            model=model,
            transform=transform,
            device=device,
            batch_size=args.batch_size,
        )
        candidates = find_cross_split_feature_candidates(
            features,
            split_by_instance,
            minimum_similarity=args.minimum_similarity,
            top_k=args.top_k,
        )
    except (DatasetConfigError, OSError, RuntimeError, TypeError, ValueError) as error:
        payload = {
            "status": "AUDIT_FAILED",
            "error_code": "FROZEN_FEATURE_AUDIT_FAILED",
            "error_type": type(error).__name__,
            "software_versions": software_versions(),
        }
        print_status(payload)
        return 2

    rows = [
        {
            "left_id": candidate.left_id,
            "right_id": candidate.right_id,
            "left_split": candidate.left_split,
            "right_split": candidate.right_split,
            "cosine_similarity": f"{candidate.cosine_similarity:.12f}",
            "selected_by_threshold": candidate.selected_by_threshold,
            "selected_by_top_k": candidate.selected_by_top_k,
            "review_decision": "",
        }
        for candidate in candidates
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    _write_candidates(args.output / "feature_duplicate_candidates.csv", rows)
    threshold_count = sum(candidate.selected_by_threshold for candidate in candidates)
    top_k_count = sum(candidate.selected_by_top_k for candidate in candidates)
    summary = {
        "schema_version": 1,
        "status": "REVIEW_REQUIRED",
        "audit_scope": "LOCAL_M1_FROZEN_FEATURE_DUPLICATE_RANKING_ONLY",
        "dataset": config.name,
        "model": "torchvision_resnet50_imagenet1k_v2_penultimate_pool",
        "weights_sha256": observed_digest,
        "weights_downloaded_by_cli": False,
        "preprocessing": "torchvision_ResNet50_Weights_DEFAULT_0.15.2",
        "device": device,
        "sample_count": len(features),
        "feature_dimension": feature_dimension,
        "minimum_similarity": args.minimum_similarity,
        "top_k": args.top_k,
        "selected_candidate_count": len(candidates),
        "selected_by_threshold_count": threshold_count,
        "selected_by_top_k_count": top_k_count,
        "maximum_similarity": (
            None if not candidates else candidates[0].cosine_similarity
        ),
        "reviewer_boundary": "CANDIDATE_ONLY_NOT_DUPLICATE_IDENTITY",
        "automatic_deletion": False,
        "split_changed": False,
        "software_versions": software_versions(),
    }
    write_json(args.output / "feature_similarity_summary.json", summary)
    print_status(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

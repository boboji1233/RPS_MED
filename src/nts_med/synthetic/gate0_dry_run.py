"""Synthetic-only end-to-end Gate 0 readiness dry-run.

This module validates software interfaces. It never reads real data and its
outputs are not scientific results.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from nts_med.concepts.burden import DR_LESION_AXES, compute_dr_burden
from nts_med.concepts.fov import FOVStatus, extract_ddr_retinal_fov
from nts_med.relations.algebra import Interval, RelationConfig, infer_relation
from nts_med.statistics.bootstrap import BootstrapConfig, stratified_cluster_resamples

SYNTHETIC_GATE0_DRY_RUN_STATUS: Final = (
    "SYNTHETIC_GATE0_DRY_RUN_NOT_SCIENTIFIC_RESULT"
)


def _synthetic_image() -> NDArray[np.uint8]:
    rows, columns = np.ogrid[:96, :96]
    retinal = ((rows - 48.0) / 43.0) ** 2 + ((columns - 48.0) / 43.0) ** 2 <= 1
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    image[retinal] = np.array([80, 30, 20], dtype=np.uint8)
    return image


def _high_masks(fov: NDArray[np.bool_]) -> dict[str, NDArray[np.uint8]]:
    high: dict[str, NDArray[np.uint8]] = {}
    ma = np.zeros(fov.shape, dtype=np.uint8)
    ma[::3, ::3] = 255
    ma[np.logical_not(fov)] = 0
    high["ma"] = ma
    rows, columns = np.indices(fov.shape)
    area = np.logical_and(fov, (rows + columns) % 4 != 0)
    for axis in ("hem", "ex", "cws"):
        high[axis] = np.where(area, 255, 0).astype(np.uint8)
    return high


def _point_intervals(values: Mapping[str, float]) -> dict[str, Interval]:
    return {axis: Interval(value, value) for axis, value in values.items()}


def _resample_digest(resamples: object) -> str:
    encoded = json.dumps(resamples, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_synthetic_gate0_dry_run(
    output_dir: str | Path,
    relation_config: RelationConfig,
    *,
    seed: int = 20260823,
    bootstrap_replicates: int = 5000,
) -> dict[str, object]:
    """Run FOV-to-relation and bootstrap interfaces on constructed arrays."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    image = _synthetic_image()
    fov_result = extract_ddr_retinal_fov(image)
    if fov_result.status is not FOVStatus.VALID or fov_result.mask is None:
        raise RuntimeError("synthetic valid FOV fixture failed")
    fov = fov_result.mask

    zero_masks = {
        axis: np.zeros(fov.shape, dtype=np.uint8) for axis in DR_LESION_AXES
    }
    high_masks = _high_masks(fov)
    minimums = {axis: 1 for axis in DR_LESION_AXES}
    low = compute_dr_burden(
        zero_masks, fov, minimums, scale=1_000_000.0, positive_class=255
    )
    high = compute_dr_burden(
        high_masks, fov, minimums, scale=1_000_000.0, positive_class=255
    )
    if any(high[axis] <= relation_config.delta[axis] for axis in DR_LESION_AXES):
        raise ValueError("synthetic burden range does not exceed configured delta")

    cross_left = {
        axis: high[axis] if axis in ("ma", "hem") else low[axis]
        for axis in DR_LESION_AXES
    }
    cross_right = {
        axis: high[axis] if axis in ("ex", "cws") else low[axis]
        for axis in DR_LESION_AXES
    }
    wide = {
        axis: Interval(0.0, relation_config.max_width[axis] + 1.0)
        for axis in DR_LESION_AXES
    }
    relation_inputs = {
        "LESS": (_point_intervals(low), _point_intervals(high)),
        "GREATER": (_point_intervals(high), _point_intervals(low)),
        "STRUCTURAL_INCOMPARABLE": (
            _point_intervals(cross_left),
            _point_intervals(cross_right),
        ),
        "EQUIVALENT": (_point_intervals(low), _point_intervals(low)),
        "UNRESOLVED": (wide, wide),
    }
    inferred = {
        expected: infer_relation(left, right, relation_config).name
        for expected, (left, right) in relation_inputs.items()
    }
    expected_match = all(expected == observed for expected, observed in inferred.items())

    bootstrap_config = BootstrapConfig(
        replicates=bootstrap_replicates,
        seed=seed,
        confidence_level=0.95,
    )
    resamples = stratified_cluster_resamples(
        {
            "synthetic_source_a": tuple(f"a{index}" for index in range(5)),
            "synthetic_source_b": tuple(f"b{index}" for index in range(7)),
        },
        bootstrap_config,
    )

    empty_fov = extract_ddr_retinal_fov(np.zeros_like(image))
    multiple_image = np.zeros_like(image)
    multiple_image[5:30, 5:30] = 80
    multiple_image[65:90, 65:90] = 80
    multiple_fov = extract_ddr_retinal_fov(multiple_image)
    illegal_mask_failed = False
    shape_mismatch_failed = False
    try:
        bad_masks = dict(zero_masks)
        bad_masks["ma"] = np.full(fov.shape, 2, dtype=np.uint8)
        compute_dr_burden(bad_masks, fov, minimums, scale=1_000_000.0)
    except ValueError:
        illegal_mask_failed = True
    try:
        bad_masks = dict(zero_masks)
        bad_masks["ma"] = np.zeros((2, 2), dtype=np.uint8)
        compute_dr_burden(
            bad_masks, fov, minimums, scale=1_000_000.0, positive_class=255
        )
    except ValueError:
        shape_mismatch_failed = True

    fail_closed = {
        "empty_fov": empty_fov.status is FOVStatus.EMPTY,
        "multiple_components": multiple_fov.status is FOVStatus.MULTIPLE_COMPONENTS,
        "illegal_mask": illegal_mask_failed,
        "shape_mismatch": shape_mismatch_failed,
    }
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": SYNTHETIC_GATE0_DRY_RUN_STATUS,
        "seed": seed,
        "real_data_used": False,
        "scientific_result": False,
        "fov": {
            "valid_fixture": fov_result.status.value,
            "empty_fixture": empty_fov.status.value,
            "multiple_component_fixture": multiple_fov.status.value,
        },
        "burden": {
            "axes": list(DR_LESION_AXES),
            "positive_class_explicit": 255,
            "density_scale": 1_000_000.0,
        },
        "relations": {
            "expected_match": expected_match,
            "counts": dict(sorted(Counter(inferred.values()).items())),
        },
        "bootstrap": {
            "method": "STRATIFIED_IMAGE_INSTANCE_CLUSTER_BOOTSTRAP",
            "replicates": bootstrap_replicates,
            "resample_sha256": _resample_digest(resamples),
        },
        "fail_closed": fail_closed,
    }
    summary["all_checks_passed"] = expected_match and all(fail_closed.values())
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


__all__ = ["SYNTHETIC_GATE0_DRY_RUN_STATUS", "run_synthetic_gate0_dry_run"]

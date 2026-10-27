"""Strict loader for the RPS-Med R1 v0.7 measurement-amendment overlay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nts_med.reproducibility.freeze import sha256_file
from rps_med.cli.formal_r1 import _mapping, _relative_file, _yaml_mapping
from rps_med.fov_v07 import (
    DDR_DOMINANT_COMPONENT_MIN_SHARE,
    DDR_FOV_V07_ALGORITHM_VERSION,
)


def load_v07_config(root: Path, overlay_path: Path) -> dict[str, Any]:
    """Validate v0.7 as one explicit FOV amendment over frozen v0.5/v0.6."""

    overlay = _yaml_mapping(overlay_path)
    allowed = {
        "schema_version",
        "config_id",
        "preregistration",
        "status",
        "scientific_base_config",
        "scientific_base_config_sha256",
        "diagnostics_base_config",
        "diagnostics_base_config_sha256",
        "prior_formal_result",
        "prior_formal_result_sha256",
        "measurement_amendment",
        "development_validation",
        "scientific_changes",
        "execution",
    }
    if set(overlay) != allowed:
        raise ValueError("R1 v0.7 overlay fields are invalid")
    if overlay.get("schema_version") != 1 or overlay.get("config_id") != "rps_med_r1_v0.7":
        raise ValueError("R1 v0.7 overlay identity is invalid")
    if overlay.get("preregistration") != "metadata/preregistration/rps_med_r1_v0.7.yaml":
        raise ValueError("R1 v0.7 preregistration binding is invalid")

    base_path = _bound_file(
        root,
        overlay,
        path_field="scientific_base_config",
        hash_field="scientific_base_config_sha256",
        expected_path="configs/experiment/rps_med_r1_v0.5.yaml",
    )
    diagnostics_path = _bound_file(
        root,
        overlay,
        path_field="diagnostics_base_config",
        hash_field="diagnostics_base_config_sha256",
        expected_path="configs/experiment/rps_med_r1_v0.6.yaml",
    )
    _bound_file(
        root,
        overlay,
        path_field="prior_formal_result",
        hash_field="prior_formal_result_sha256",
        expected_path="metadata/gates/rps_med_r1_v06_formal_result_2026-08-25_v1.yaml",
    )

    measurement = _mapping(overlay.get("measurement_amendment"), name="measurement_amendment")
    expected_measurement = {
        "algorithm_version": DDR_FOV_V07_ALGORITHM_VERSION,
        "behavior_change_limited_to": "LEGACY_MULTIPLE_COMPONENTS",
        "background_threshold": 5.0,
        "minimum_area_ratio": 0.10,
        "maximum_area_ratio": 0.98,
        "substantial_component_ratio": 0.001,
        "dominant_component_min_share": DDR_DOMINANT_COMPONENT_MIN_SHARE,
        "dominance_denominator": "ALL_SUBSTANTIAL_FOREGROUND_PIXELS",
        "unique_largest_required": True,
        "disconnected_component_union": "FORBIDDEN",
        "rectangular_image_denominator": "FORBIDDEN",
        "lesion_mask_or_grade_conditioning": "FORBIDDEN",
        "ambiguous_multiple_components": "TECHNICAL_ABORT",
        "zero_exclusion_contract": "REQUIRED",
    }
    if dict(measurement) != expected_measurement:
        raise ValueError("R1 v0.7 measurement amendment is invalid")

    development = _mapping(overlay.get("development_validation"), name="development_validation")
    expected_development = {
        "scope": "DDR_LESION_OFFICIAL_TRAIN",
        "expected_input_count": 383,
        "image_decode_exclusion_count_required": 0,
        "legacy_valid_mask_difference_count_required": 0,
        "lesion_masks_opened": False,
        "relations_computed": False,
        "aggregate_output_only": True,
    }
    if dict(development) != expected_development:
        raise ValueError("R1 v0.7 development validation contract is invalid")

    changes = _mapping(overlay.get("scientific_changes"), name="scientific_changes")
    expected_changes = {
        "fov_measurement_algorithm": True,
        "source_or_boundary": False,
        "lesion_axis_semantics": False,
        "burden_formula_except_fov_mask": False,
        "relation_algebra": False,
        "reference_thresholds": False,
        "tolerance_scale_grid": False,
        "bootstrap": False,
        "decision_thresholds": False,
    }
    if dict(changes) != expected_changes:
        raise ValueError("R1 v0.7 scientific-change disclosure is invalid")

    execution = _mapping(overlay.get("execution"), name="execution")
    expected_execution = {
        "formal_r1": "NOT_AUTHORIZED",
        "model_training": "FORBIDDEN",
        "gate0_rerun": "FORBIDDEN",
        "prior_authorization_reuse": "FORBIDDEN",
        "output_overwrite": "FORBIDDEN",
        "result_feedback_tuning": "FORBIDDEN",
        "automatic_fix": "FORBIDDEN",
        "automatic_retry": "FORBIDDEN",
    }
    if dict(execution) != expected_execution:
        raise ValueError("R1 v0.7 execution contract is invalid")

    diagnostics = _yaml_mapping(diagnostics_path)
    base = _yaml_mapping(base_path)
    merged = dict(base)
    merged.update(
        {
            "config_id": "rps_med_r1_v0.7",
            "preregistration": overlay["preregistration"],
            "status": overlay["status"],
            "diagnostics": diagnostics["diagnostics"],
            "measurement_amendment": dict(measurement),
            "development_validation": dict(development),
            "scientific_changes": dict(changes),
            "execution": dict(execution),
        }
    )
    return merged


def _bound_file(
    root: Path,
    overlay: dict[str, Any],
    *,
    path_field: str,
    hash_field: str,
    expected_path: str,
) -> Path:
    if overlay.get(path_field) != expected_path:
        raise ValueError(f"R1 v0.7 {path_field} binding is invalid")
    path = _relative_file(root, overlay[path_field], name=path_field)
    if sha256_file(path) != overlay.get(hash_field):
        raise ValueError(f"R1 v0.7 {path_field} hash mismatch")
    return path


__all__ = ["load_v07_config"]

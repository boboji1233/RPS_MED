from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from rps_med.cli import development_fov_v07, synthetic_r1_v07
from rps_med.config_v07 import load_v07_config


def test_v07_config_discloses_only_fov_measurement_change(tmp_path: Path) -> None:
    root = Path.cwd()
    config = load_v07_config(root, root / "configs/experiment/rps_med_r1_v0.7.yaml")
    changes = config["scientific_changes"]
    assert changes["fov_measurement_algorithm"] is True
    assert all(
        value is False
        for key, value in changes.items()
        if key != "fov_measurement_algorithm"
    )

    overlay = yaml.safe_load(
        (root / "configs/experiment/rps_med_r1_v0.7.yaml").read_text(encoding="utf-8")
    )
    overlay["measurement_amendment"]["dominant_component_min_share"] = 0.80
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="measurement amendment"):
        load_v07_config(root, path)


def test_v07_synthetic_output_is_byte_deterministic(tmp_path: Path) -> None:
    root = Path.cwd()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    common = [
        "--repository-root",
        str(root),
        "--config",
        str(root / "configs/experiment/rps_med_r1_v0.7.yaml"),
    ]
    assert synthetic_r1_v07.main([*common, "--output", str(first)]) == 0
    assert synthetic_r1_v07.main([*common, "--output", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert all(payload["checks"].values())
    assert payload["scientific_data_opened"] is False
    assert payload["formal_r1"] == "NOT_RUN_SYNTHETIC_ONLY"


def test_v07_development_cli_blocks_failed_invariance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("synthetic\n", encoding="utf-8")
    output = tmp_path / "output.json"
    monkeypatch.setattr(development_fov_v07, "sha256_file", lambda *args: "expected")
    monkeypatch.setattr(
        development_fov_v07,
        "load_v07_config",
        lambda *args: {
            "input_identity": {"ddr_private_manifest": "expected"},
            "development_validation": {
                "expected_input_count": 383,
                "image_decode_exclusion_count_required": 0,
                "legacy_valid_mask_difference_count_required": 0,
            },
            "measurement_amendment": {"dominant_component_min_share": 0.90},
        },
    )
    monkeypatch.setattr(
        development_fov_v07,
        "audit_ddr_train_fov_v07",
        lambda *args, **kwargs: {
            "observed_input_count": 383,
            "image_decode_exclusion_count": 0,
            "legacy_valid_mask_difference_count": 1,
            "lesion_masks_opened": False,
            "relations_computed": False,
        },
    )
    arguments = [
        "--repository-root",
        str(tmp_path),
        "--config",
        str(tmp_path / "config.yaml"),
        "--ddr-root",
        str(tmp_path),
        "--private-manifest",
        str(manifest),
        "--output",
        str(output),
    ]
    assert development_fov_v07.main(arguments) == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "RPS_MED_R1_V0_7_DEVELOPMENT_FOV_AUDIT_BLOCKED"
    assert payload["checks"]["legacy_mask_invariance"] is False
    assert payload["formal_r1"] == "NOT_RUN"

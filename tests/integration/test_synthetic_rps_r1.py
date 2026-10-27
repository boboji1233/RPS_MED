from __future__ import annotations

import json
from pathlib import Path

import yaml

from rps_med.cli.synthetic_r1 import main


def test_synthetic_r1_controls_are_deterministic_and_identifier_free(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(
        Path("configs/experiment/rps_med_r1_v0.5.yaml").read_text(encoding="utf-8")
    )
    config["bootstrap"]["replicates"] = 100
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    assert main(["--config", str(config_path), "--output", str(first)]) == 0
    assert main(["--config", str(config_path), "--output", str(second)]) == 0

    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["positive_control"]["decision"] == "GO"
    assert payload["negative_control"]["decision"] == "NO_GO"
    assert payload["formal_r1"] == "NOT_RUN_SYNTHETIC_ONLY"
    assert "synthetic_maples" not in first.read_text(encoding="utf-8")

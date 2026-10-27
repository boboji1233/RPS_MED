from __future__ import annotations

import json
from pathlib import Path

from rps_med.cli.synthetic_r1_v06 import main


def test_v06_synthetic_diagnostics_are_deterministic_and_private(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    arguments = [
        "--repository-root",
        str(Path.cwd()),
        "--config",
        str(Path("configs/experiment/rps_med_r1_v0.6.yaml")),
    ]
    assert main([*arguments, "--output", str(first)]) == 0
    assert main([*arguments, "--output", str(second)]) == 0

    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert len(payload["substage_fault_matrix"]) == 8
    assert payload["exclusion_reason_matrix"]["measurement_exclusion_count"] == 10
    assert payload["formal_r1"] == "NOT_RUN_SYNTHETIC_ONLY"
    assert payload["scientific_data_opened"] is False
    assert "synthetic detail" not in first.read_text(encoding="utf-8")

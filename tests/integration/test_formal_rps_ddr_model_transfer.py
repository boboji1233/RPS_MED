from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rps_med.cli import formal_ddr_model_transfer


def test_formal_ddr_transfer_claims_before_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {
        name: tmp_path / name
        for name in (
            "passport.yaml",
            "authorization.yaml",
            "manifest.csv",
            "cache.npz",
            "config.yaml",
            "development.yaml",
        )
    }
    for path in files.values():
        path.write_text("synthetic\n", encoding="utf-8")
    weights = tmp_path / "weights"
    weights.mkdir()
    ddr = tmp_path / "ddr"
    ddr.mkdir()
    marker = tmp_path / "marker.json"
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        formal_ddr_model_transfer,
        "validate_freeze_passport",
        lambda *args, **kwargs: SimpleNamespace(
            valid=True, freeze_id="rps_med_ddr_model_transfer_v0.1_synthetic"
        ),
    )
    monkeypatch.setattr(formal_ddr_model_transfer, "_git_freeze_valid", lambda *args: True)
    monkeypatch.setattr(formal_ddr_model_transfer, "_interpreter_valid", lambda *args: True)
    monkeypatch.setattr(formal_ddr_model_transfer, "sha256_file", lambda *args: "a" * 64)
    monkeypatch.setattr(
        formal_ddr_model_transfer,
        "load_yaml_mapping",
        lambda *args: {"input_files": {}},
    )
    monkeypatch.setattr(
        formal_ddr_model_transfer,
        "_input_file_checks",
        lambda *args: {"all": True},
    )
    def _claim(*args: object, **kwargs: object) -> SimpleNamespace:
        marker.write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(authorization_sha256="b" * 64, marker_sha256="c" * 64)

    monkeypatch.setattr(formal_ddr_model_transfer, "claim_one_run_authorization", _claim)
    monkeypatch.setattr(
        formal_ddr_model_transfer,
        "validate_ddr_confirmation_manifest",
        lambda *args: {"valid": True},
    )

    def _evaluate(*args: object, **kwargs: object) -> dict[str, object]:
        assert marker.is_file()
        return {"status": "RPS_MED_DDR_MODEL_TRANSFER_COMPLETE", "decision": "TRANSFER_PASS"}

    monkeypatch.setattr(formal_ddr_model_transfer, "evaluate_ddr_model_transfer", _evaluate)
    arguments = [
        "--mode", "run",
        "--repository-root", str(tmp_path),
        "--passport", str(files["passport.yaml"]),
        "--authorization", str(files["authorization.yaml"]),
        "--consumption-marker", str(marker),
        "--ddr-manifest", str(files["manifest.csv"]),
        "--train-cache", str(files["cache.npz"]),
        "--config", str(files["config.yaml"]),
        "--development-config", str(files["development.yaml"]),
        "--weights-directory", str(weights),
        "--ddr-root", str(ddr),
        "--output", str(output),
    ]
    assert formal_ddr_model_transfer.main(arguments) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["decision"] == "TRANSFER_PASS"
    assert payload["authorization_consumed"] is True

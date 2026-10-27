from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rps_med.cli import formal_tjdr_model_test


def _arguments(tmp_path: Path) -> tuple[list[str], Path, Path]:
    files = {
        name: tmp_path / name
        for name in ("passport.yaml", "authorization.yaml", "cache.npz", "test.zip", "config.yaml")
    }
    for path in files.values():
        path.write_text("synthetic fixture\n", encoding="utf-8")
    weights = tmp_path / "weights"
    weights.mkdir()
    output = tmp_path / "output.json"
    marker = tmp_path / "marker.json"
    return (
        [
            "--repository-root",
            str(tmp_path),
            "--passport",
            str(files["passport.yaml"]),
            "--authorization",
            str(files["authorization.yaml"]),
            "--consumption-marker",
            str(marker),
            "--train-cache",
            str(files["cache.npz"]),
            "--test-archive",
            str(files["test.zip"]),
            "--config",
            str(files["config.yaml"]),
            "--weights-directory",
            str(weights),
            "--output",
            str(output),
        ],
        output,
        marker,
    )


def test_formal_model_claims_before_scientific_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, output, marker = _arguments(tmp_path)
    monkeypatch.setattr(formal_tjdr_model_test, "sha256_file", lambda *args: "a" * 64)
    monkeypatch.setattr(
        formal_tjdr_model_test,
        "load_yaml_mapping",
        lambda *args: {"freeze_id": "synthetic", "input_files": {}},
    )
    monkeypatch.setattr(
        formal_tjdr_model_test,
        "validate_freeze_passport",
        lambda *args, **kwargs: SimpleNamespace(valid=True),
    )
    monkeypatch.setattr(
        formal_tjdr_model_test,
        "validate_authorization",
        lambda *args, **kwargs: (True, ()),
    )
    monkeypatch.setattr(
        formal_tjdr_model_test,
        "_input_identity",
        lambda *args, **kwargs: {"all": True},
    )

    def synthetic_evaluation(*args: object, **kwargs: object) -> dict[str, object]:
        assert marker.is_file()
        return {
            "status": "RPS_MED_TJDR_MODEL_FORMAL_TEST_COMPLETE",
            "decision": "PASS",
            "row_predictions_persisted": False,
        }

    monkeypatch.setattr(
        formal_tjdr_model_test,
        "evaluate_formal_tjdr_model_test",
        synthetic_evaluation,
    )

    assert formal_tjdr_model_test.main(arguments) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["decision"] == "PASS"
    assert payload["authorization_consumed"] is True
    assert payload["row_predictions_persisted"] is False

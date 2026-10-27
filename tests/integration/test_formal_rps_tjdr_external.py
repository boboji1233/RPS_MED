from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nts_med.gate0.execution import MeasurementRecord, SourceMeasurements
from rps_med.cli import formal_tjdr_external


def _source(name: str, role: str, count: int) -> SourceMeasurements:
    records = tuple(
        MeasurementRecord(
            instance_id=f"opaque-{name}-{index}",
            primary={axis: float(index) for axis in ("ma", "hem", "ex", "cws")},
            variants={},
            grade=None,
            grading_split_conflict=False,
            strict_measurement_quality=True,
        )
        for index in range(count)
    )
    return SourceMeasurements(name, role, count, count, records, 0)


def _arguments(tmp_path: Path, *, mode: str) -> tuple[list[str], Path]:
    paths = {
        name: tmp_path / name
        for name in (
            "passport.yaml",
            "config.yaml",
            "authorization.yaml",
            "train.zip",
            "test.zip",
            "development.json",
        )
    }
    for path in paths.values():
        path.write_text("synthetic non-scientific fixture\n", encoding="utf-8")
    output = tmp_path / "output.json"
    marker = tmp_path / "marker.json"
    return (
        [
            "--mode",
            mode,
            "--repository-root",
            str(tmp_path),
            "--passport",
            str(paths["passport.yaml"]),
            "--authorization",
            str(paths["authorization.yaml"]),
            "--consumption-marker",
            str(marker),
            "--config",
            str(paths["config.yaml"]),
            "--train-archive",
            str(paths["train.zip"]),
            "--test-archive",
            str(paths["test.zip"]),
            "--development-reference",
            str(paths["development.json"]),
            "--output",
            str(output),
        ],
        output,
    )


def _patch_common(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        formal_tjdr_external,
        "validate_freeze_passport",
        lambda *args, **kwargs: SimpleNamespace(
            valid=True, freeze_id="rps_med_tjdr_external_gate_v0.1_synthetic"
        ),
    )
    monkeypatch.setattr(formal_tjdr_external, "_git_freeze_valid", lambda *args: True)
    monkeypatch.setattr(formal_tjdr_external, "_interpreter_valid", lambda *args: True)
    monkeypatch.setattr(
        formal_tjdr_external,
        "_contract",
        lambda *args: {
            "source": {"expected_development_count": 3, "expected_test_count": 2},
            "annotation": {"allow_all_background_complete_annotation": True},
            "fov": {
                "background_threshold": 5.0,
                "minimum_area_ratio": 0.1,
                "maximum_area_ratio": 0.98,
                "substantial_component_ratio": 0.001,
                "dominant_component_min_share": 0.9,
            },
        },
    )


def test_external_preflight_does_not_claim_or_open_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, output = _arguments(tmp_path, mode="preflight")
    _patch_common(monkeypatch)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("preflight must not claim authorization or open science")

    monkeypatch.setattr(formal_tjdr_external, "claim_one_run_authorization", forbidden)
    monkeypatch.setattr(formal_tjdr_external, "load_tjdr_train_measurements", forbidden)
    monkeypatch.setattr(formal_tjdr_external, "load_tjdr_test_measurements", forbidden)

    assert formal_tjdr_external.main(arguments) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "RPS_MED_TJDR_EXTERNAL_PREFLIGHT_PASS_AWAITING_AUTHORIZATION"
    assert payload["authorization_claimed"] is False
    assert payload["test_scientific_data_opened"] is False


def test_external_run_claims_before_test_and_persists_aggregate_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, output = _arguments(tmp_path, mode="run")
    _patch_common(monkeypatch)
    events: list[str] = []
    monkeypatch.setattr(
        formal_tjdr_external,
        "claim_one_run_authorization",
        lambda *args, **kwargs: (
            events.append("claim")
            or SimpleNamespace(authorization_sha256="a" * 64, marker_sha256="b" * 64)
        ),
    )
    monkeypatch.setattr(formal_tjdr_external, "_identity", lambda *args: {})
    monkeypatch.setattr(
        formal_tjdr_external,
        "load_tjdr_train_measurements",
        lambda *args, **kwargs: _source("train", "DEVELOPMENT_CALIBRATION", 3),
    )
    monkeypatch.setattr(
        formal_tjdr_external, "_development_reference_matches", lambda *args: True
    )

    def test_source(*args: object, **kwargs: object) -> SourceMeasurements:
        assert events == ["claim"]
        events.append("test")
        return _source("test", "INDEPENDENT_EXTERNAL_REPLICATION", 2)

    monkeypatch.setattr(formal_tjdr_external, "load_tjdr_test_measurements", test_source)
    monkeypatch.setattr(
        formal_tjdr_external,
        "execute_tjdr_external_gate",
        lambda *args: {
            "schema_version": 1,
            "status": "FORMAL_RPS_MED_TJDR_EXTERNAL_GATE_COMPLETE",
            "decision": "GO",
        },
    )

    assert formal_tjdr_external.main(arguments) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert events == ["claim", "test"]
    assert payload["decision"] == "GO"
    assert payload["authorization_consumed"] is True
    assert payload["test_opened_after_authorization_claim"] is True
    assert payload["raw_data_persisted"] is False

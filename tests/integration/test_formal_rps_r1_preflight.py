from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from rps_med.cli import formal_r1


def _write(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[list[str], Path]:
    config_path = tmp_path / "configs" / "r1.yaml"
    config_hash = _write(
        config_path,
        yaml.safe_dump(
            {
                "config_id": "rps_med_r1_v0.5",
                "execution": {
                    "formal_r1": "NOT_AUTHORIZED",
                    "model_training": "FORBIDDEN",
                },
                "confirmation_boundary": {
                    "maples_development_count": 2,
                    "maples_confirmation_count": 2,
                    "ddr_confirmation_count": 1,
                },
            }
        ),
    )
    private_manifest = tmp_path / "private.csv"
    manifest_hash = _write(private_manifest, "private manifest identity\n")
    maples = tmp_path / "maples.zip"
    additional = tmp_path / "additional.zip"
    _write(maples, "must not open")
    _write(additional, "must not open")
    ddr = tmp_path / "ddr"
    ddr.mkdir()
    passport = tmp_path / "passport.yaml"
    passport.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "freeze_id": "rps_med_r1_v0.5_synthetic",
                "status": "FROZEN_READY_AWAITING_AUTHORIZATION",
                "implementation_commit": "0" * 40,
                "execution": {
                    "gate0": "NOT_RUN",
                    "formal_r1": "NOT_RUN",
                    "training": "FORBIDDEN",
                    "authorization_required": True,
                },
                "runner": {"entry_point": "rps-med-formal-r1"},
                "artifacts": {
                    "experiment_config": {
                        "path": "configs/r1.yaml",
                        "sha256": config_hash,
                    }
                },
                "input_manifests": {"ddr": manifest_hash},
                "input_files": {"maples": "a" * 64},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return (
        [
            "--repository-root",
            str(tmp_path),
            "--passport",
            str(passport),
            "--private-manifest",
            f"ddr={private_manifest}",
            "--config",
            str(config_path),
            "--maples-archive",
            str(maples),
            "--maples-additional",
            str(additional),
            "--ddr-root",
            str(ddr),
        ],
        passport,
    )


def test_formal_r1_preflight_does_not_open_scientific_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common, _ = _fixture(tmp_path)
    monkeypatch.setattr(formal_r1, "_git_freeze_valid", lambda *_: True)
    monkeypatch.setattr(formal_r1, "_interpreter_valid", lambda *_: True)
    output = tmp_path / "preflight.json"

    status = formal_r1.main(
        [
            "--mode",
            "preflight",
            *common,
            "--output",
            str(output),
        ]
    )

    assert status == 0
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert payload["scientific_data_opened"] is False
    assert payload["raw_input_bytes_opened"] is False
    assert payload["formal_r1"] == "NOT_RUN"


def test_formal_r1_post_authorization_failure_is_stable_and_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common, passport = _fixture(tmp_path)
    monkeypatch.setattr(formal_r1, "_git_freeze_valid", lambda *_: True)
    monkeypatch.setattr(formal_r1, "_interpreter_valid", lambda *_: True)
    monkeypatch.setattr(
        formal_r1,
        "_verify_identity",
        lambda *_, **__: {"checks": {"synthetic": True}},
    )

    def fail_measurement(*_: object, **__: object) -> object:
        raise ValueError("private detail must not be recorded")

    monkeypatch.setattr(
        formal_r1, "load_maples_confirmation_measurements", fail_measurement
    )
    authorization = tmp_path / "authorization.yaml"
    authorization.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "authorization": "FORMAL_R1_ONCE",
                "authorized": True,
                "freeze_id": "rps_med_r1_v0.5_synthetic",
                "freeze_passport_sha256": hashlib.sha256(
                    passport.read_bytes()
                ).hexdigest(),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    marker = tmp_path / "claimed.json"
    output = tmp_path / "aborted.json"

    status = formal_r1.main(
        [
            "--mode",
            "run",
            *common,
            "--authorization",
            str(authorization),
            "--consumption-marker",
            str(marker),
            "--output",
            str(output),
        ]
    )

    assert status == 2
    assert marker.is_file()
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert payload["status"] == "FORMAL_RPS_MED_R1_ABORTED_TECHNICAL"
    assert payload["failed_stage"] == "MAPLES_CONFIRMATION_MEASUREMENT"
    assert payload["error_family"] == "VALUE_ERROR"
    assert "private detail" not in output.read_text(encoding="utf-8")
    assert payload["automatic_retry"] == "FORBIDDEN"

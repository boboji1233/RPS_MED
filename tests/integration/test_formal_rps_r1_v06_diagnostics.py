from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from nts_med.gate0.execution import MeasurementRecord, SourceMeasurements
from rps_med.cli import formal_r1_v06
from rps_med.source_io_v06 import (
    DDRDiagnosticState,
    DDRExclusionReason,
    DDRSourceMeasurementError,
    DDRSubstage,
)


def _source(name: str, count: int) -> SourceMeasurements:
    records = tuple(
        MeasurementRecord(
            instance_id=f"opaque-{index}",
            primary={axis: 0.0 for axis in ("ma", "hem", "ex", "cws")},
            variants={},
            grade=None,
            grading_split_conflict=False,
            strict_measurement_quality=True,
        )
        for index in range(count)
    )
    return SourceMeasurements(name, "FORMAL_R1_CONFIRMATION", count, count, records, 0)


def _arguments(tmp_path: Path, *, mode: str) -> tuple[list[str], Path]:
    passport = tmp_path / "passport.yaml"
    config = tmp_path / "config.yaml"
    authorization = tmp_path / "authorization.yaml"
    marker = tmp_path / "marker.json"
    manifest = tmp_path / "manifest.csv"
    maples = tmp_path / "maples.zip"
    additional = tmp_path / "additional.zip"
    ddr = tmp_path / "ddr"
    output = tmp_path / "output.json"
    for path in (passport, config, authorization, manifest, maples, additional):
        path.write_text("synthetic non-scientific fixture\n", encoding="utf-8")
    ddr.mkdir()
    return (
        [
            "--mode",
            mode,
            "--repository-root",
            str(tmp_path),
            "--passport",
            str(passport),
            "--authorization",
            str(authorization),
            "--consumption-marker",
            str(marker),
            "--private-manifest",
            f"ddr={manifest}",
            "--config",
            str(config),
            "--maples-archive",
            str(maples),
            "--maples-additional",
            str(additional),
            "--ddr-root",
            str(ddr),
            "--output",
            str(output),
        ],
        output,
    )


def _patch_common(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        formal_r1_v06,
        "validate_freeze_passport",
        lambda *args, **kwargs: SimpleNamespace(valid=True, freeze_id="rps_med_r1_v0.6_synthetic"),
    )
    monkeypatch.setattr(formal_r1_v06, "_git_freeze_valid", lambda *args: True)
    monkeypatch.setattr(formal_r1_v06, "_interpreter_valid", lambda *args: True)
    monkeypatch.setattr(
        formal_r1_v06,
        "_v06_passport_contract",
        lambda *args: {
            "confirmation_boundary": {
                "maples_development_count": 2,
                "maples_confirmation_count": 2,
                "ddr_confirmation_count": 1,
            }
        },
    )
    monkeypatch.setattr(formal_r1_v06, "_verify_identity", lambda *args, **kwargs: {"checks": {}})
    monkeypatch.setattr(
        formal_r1_v06,
        "claim_one_run_authorization",
        lambda *args, **kwargs: SimpleNamespace(
            authorization_sha256="a" * 64, marker_sha256="b" * 64
        ),
    )


def test_v06_preflight_does_not_claim_or_open_scientific_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, output = _arguments(tmp_path, mode="preflight")
    _patch_common(monkeypatch)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("scientific access forbidden in preflight")

    monkeypatch.setattr(formal_r1_v06, "claim_one_run_authorization", forbidden)
    monkeypatch.setattr(formal_r1_v06, "load_maples_confirmation_measurements", forbidden)
    monkeypatch.setattr(formal_r1_v06, "load_ddr_confirmation_measurements_v06", forbidden)

    assert formal_r1_v06.main(arguments) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "RPS_MED_R1_V0_6_PREFLIGHT_PASS_AWAITING_EXACT_HASH_AUTHORIZATION"
    assert payload["authorization_claimed"] is False
    assert payload["raw_input_bytes_opened"] is False
    assert payload["scientific_data_opened"] is False
    assert payload["formal_r1"] == "NOT_RUN"


def test_v06_persists_ddr_substage_and_aggregate_exclusion_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, output = _arguments(tmp_path, mode="run")
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        formal_r1_v06,
        "load_maples_confirmation_measurements",
        lambda *args, **kwargs: _source("maples_dr_messidor1", 2),
    )
    state = DDRDiagnosticState(observed_input_count=1, eligible_count=0)
    state.exclude(DDRExclusionReason.FOV_EMPTY)
    error = DDRSourceMeasurementError(
        substage=DDRSubstage.SOURCE_SUMMARY,
        error_family="MEASUREMENT_EXCLUSION",
        diagnostics=state.safe_diagnostics(),
    )

    def fail_ddr(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(formal_r1_v06, "load_ddr_confirmation_measurements_v06", fail_ddr)

    assert formal_r1_v06.main(arguments) == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "FORMAL_RPS_MED_R1_V0_6_ABORTED_TECHNICAL"
    assert payload["failed_stage"] == "DDR_CONFIRMATION_MEASUREMENT"
    assert payload["failed_substage"] == "DDR_SOURCE_SUMMARY"
    assert payload["reason_codes"] == [
        "TECHNICAL_ERROR:DDR_CONFIRMATION_MEASUREMENT:DDR_SOURCE_SUMMARY:MEASUREMENT_EXCLUSION"
    ]
    diagnostics = payload["diagnostics"]["ddr_source_measurement"]
    assert diagnostics["exclusion_reason_counts"] == {"DDR_FOV_EMPTY": 1}
    assert diagnostics["exception_message_recorded"] is False
    assert diagnostics["source_identifiers_recorded"] is False
    assert payload["formal_r1"] == "NOT_ADJUDICATED"
    assert payload["automatic_retry"] == "FORBIDDEN"


def test_v06_overlay_inherits_v05_science_and_rejects_extra_fields(tmp_path: Path) -> None:
    root = Path.cwd()
    merged = formal_r1_v06._load_v06_config(root, root / "configs/experiment/rps_med_r1_v0.6.yaml")
    base = yaml.safe_load(
        (root / "configs/experiment/rps_med_r1_v0.5.yaml").read_text(encoding="utf-8")
    )
    for field in (
        "confirmation_boundary",
        "input_identity",
        "reference_thresholds",
        "scale",
        "bootstrap",
        "criteria",
    ):
        assert merged[field] == base[field]

    overlay = yaml.safe_load(
        (root / "configs/experiment/rps_med_r1_v0.6.yaml").read_text(encoding="utf-8")
    )
    overlay["criteria"] = {"forbidden": True}
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="overlay fields"):
        formal_r1_v06._load_v06_config(root, path)

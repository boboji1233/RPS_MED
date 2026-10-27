from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nts_med.gate0.execution import MeasurementRecord, SourceMeasurements
from rps_med.cli import formal_r1_v07
from rps_med.source_io_v06 import (
    DDRExclusionReason,
    DDRMeasurementOutcome,
    DDRSourceMeasurementError,
    DDRSubstage,
)
from rps_med.source_io_v07 import DDRDiagnosticStateV07


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
    paths = {
        name: tmp_path / name
        for name in (
            "passport.yaml",
            "config.yaml",
            "authorization.yaml",
            "manifest.csv",
            "maples.zip",
            "additional.zip",
        )
    }
    for path in paths.values():
        path.write_text("synthetic non-scientific fixture\n", encoding="utf-8")
    ddr = tmp_path / "ddr"
    ddr.mkdir()
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
            "--private-manifest",
            f"ddr={paths['manifest.csv']}",
            "--config",
            str(paths["config.yaml"]),
            "--maples-archive",
            str(paths["maples.zip"]),
            "--maples-additional",
            str(paths["additional.zip"]),
            "--ddr-root",
            str(ddr),
            "--output",
            str(output),
        ],
        output,
    )


def _patch_common(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        formal_r1_v07,
        "validate_freeze_passport",
        lambda *args, **kwargs: SimpleNamespace(
            valid=True, freeze_id="rps_med_r1_v0.7_synthetic"
        ),
    )
    monkeypatch.setattr(formal_r1_v07, "_git_freeze_valid", lambda *args: True)
    monkeypatch.setattr(formal_r1_v07, "_interpreter_valid", lambda *args: True)
    monkeypatch.setattr(
        formal_r1_v07,
        "_v07_passport_contract",
        lambda *args: {
            "confirmation_boundary": {
                "maples_development_count": 2,
                "maples_confirmation_count": 2,
                "ddr_confirmation_count": 1,
            },
            "measurement_amendment": {"dominant_component_min_share": 0.90},
        },
    )
    monkeypatch.setattr(formal_r1_v07, "_verify_identity", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        formal_r1_v07,
        "claim_one_run_authorization",
        lambda *args, **kwargs: SimpleNamespace(
            authorization_sha256="a" * 64, marker_sha256="b" * 64
        ),
    )


def test_v07_preflight_does_not_claim_or_open_scientific_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, output = _arguments(tmp_path, mode="preflight")
    _patch_common(monkeypatch)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("scientific access forbidden in preflight")

    monkeypatch.setattr(formal_r1_v07, "claim_one_run_authorization", forbidden)
    monkeypatch.setattr(formal_r1_v07, "load_maples_confirmation_measurements", forbidden)
    monkeypatch.setattr(formal_r1_v07, "load_ddr_confirmation_measurements_v07", forbidden)

    assert formal_r1_v07.main(arguments) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == (
        "RPS_MED_R1_V0_7_PREFLIGHT_PASS_AWAITING_EXACT_HASH_AUTHORIZATION"
    )
    assert payload["authorization_claimed"] is False
    assert payload["scientific_data_opened"] is False
    assert payload["formal_r1"] == "NOT_RUN"


def test_v07_formal_path_uses_frozen_amendment_and_persists_aggregate_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, output = _arguments(tmp_path, mode="run")
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        formal_r1_v07,
        "load_maples_confirmation_measurements",
        lambda *args, **kwargs: _source("maples_dr_messidor1", 2),
    )
    diagnostics = {
        "fov_resolution_counts": {"DDR_FOV_DOMINANT_COMPONENT_ACCEPTED": 1},
        "source_identifiers_recorded": False,
    }
    monkeypatch.setattr(
        formal_r1_v07,
        "load_ddr_confirmation_measurements_v07",
        lambda *args, **kwargs: DDRMeasurementOutcome(_source("ddr", 1), diagnostics),
    )
    monkeypatch.setattr(
        formal_r1_v07,
        "execute_formal_r1",
        lambda *args: {"schema_version": 1, "decision": "GO"},
    )

    assert formal_r1_v07.main(arguments) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "FORMAL_RPS_MED_R1_V0_7_COMPLETE"
    assert payload["measurement_amendment"]["dominant_component_min_share"] == 0.90
    assert payload["diagnostics"]["ddr_source_measurement"] == diagnostics
    assert payload["authorization_consumed"] is True
    assert payload["raw_data_persisted"] is False


def test_v07_unresolved_multiple_component_remains_technical_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, output = _arguments(tmp_path, mode="run")
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        formal_r1_v07,
        "load_maples_confirmation_measurements",
        lambda *args, **kwargs: _source("maples_dr_messidor1", 2),
    )
    state = DDRDiagnosticStateV07(observed_input_count=1, eligible_count=0)
    state.exclude(DDRExclusionReason.FOV_MULTIPLE_COMPONENTS)
    error = DDRSourceMeasurementError(
        substage=DDRSubstage.SOURCE_SUMMARY,
        error_family="MEASUREMENT_EXCLUSION",
        diagnostics=state.safe_diagnostics(),
    )

    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(formal_r1_v07, "load_ddr_confirmation_measurements_v07", fail)

    assert formal_r1_v07.main(arguments) == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "FORMAL_RPS_MED_R1_V0_7_ABORTED_TECHNICAL"
    assert payload["failed_substage"] == "DDR_SOURCE_SUMMARY"
    assert payload["formal_r1"] == "NOT_ADJUDICATED"
    assert payload["automatic_fix"] == "FORBIDDEN"

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nts_med.concepts.fov import FOVResult, FOVStatus
from nts_med.data.adapters import AdapterStatus, DatasetSample, ValidationResult
from rps_med import source_io, source_io_v06
from rps_med.source_io_v06 import (
    DDRDiagnosticState,
    DDRExclusionReason,
    DDRSourceMeasurementError,
    DDRSubstage,
)


def _sample(tmp_path: Path, *, identifier: str = "opaque-synthetic") -> DatasetSample:
    image = tmp_path / "synthetic-image.png"
    masks = {axis: tmp_path / f"synthetic-{axis}.png" for axis in ("ma", "hem", "ex", "cws")}
    return DatasetSample(
        sample_id=identifier,
        image_path=image,
        dataset_name="ddr",
        official_split="valid",
        lesion_masks=masks,
    )


def _valid_fov() -> FOVResult:
    mask = np.ones((4, 4), dtype=np.bool_)
    return FOVResult(FOVStatus.VALID, mask, 1, 1, 1.0, True)


@pytest.mark.parametrize("failed_substage", list(DDRSubstage))
def test_every_ddr_substage_discards_fault_detail(failed_substage: DDRSubstage) -> None:
    state = DDRDiagnosticState()

    def fail() -> None:
        raise ValueError("private path and source identifier must disappear")

    with pytest.raises(DDRSourceMeasurementError) as captured:
        source_io_v06._run_substage(state, failed_substage, fail)

    error = captured.value
    assert error.substage is failed_substage
    assert error.error_family == "VALUE_ERROR"
    encoded = json.dumps(error.diagnostics, sort_keys=True)
    assert "private path" not in encoded
    assert "source identifier" not in encoded
    assert error.diagnostics["absolute_paths_recorded"] is False
    assert error.diagnostics["source_identifiers_recorded"] is False


@pytest.mark.parametrize(
    ("result", "reason"),
    [
        (FOVResult(FOVStatus.EMPTY, None, 0, 0, None, False), DDRExclusionReason.FOV_EMPTY),
        (
            FOVResult(FOVStatus.MULTIPLE_COMPONENTS, None, 2, 2, None, False),
            DDRExclusionReason.FOV_MULTIPLE_COMPONENTS,
        ),
        (
            FOVResult(FOVStatus.AREA_RATIO_OUT_OF_RANGE, None, 1, 1, 0.01, False),
            DDRExclusionReason.FOV_AREA_RATIO_OUT_OF_RANGE,
        ),
        (
            FOVResult(FOVStatus.VALID, None, 1, 1, 0.5, False),
            DDRExclusionReason.FOV_OTHER_INVALID,
        ),
        (
            FOVResult(FOVStatus.VALID, np.ones((2, 2), dtype=np.bool_), 1, 1, float("nan"), False),
            DDRExclusionReason.FOV_RATIO_NONFINITE,
        ),
    ],
)
def test_fov_failures_have_fixed_aggregate_reasons(
    result: FOVResult, reason: DDRExclusionReason
) -> None:
    assert source_io_v06._invalid_fov_reason(result) is reason


def test_decode_extract_mask_burden_and_record_faults_are_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = _sample(tmp_path)

    state = DDRDiagnosticState()
    monkeypatch.setattr(
        source_io_v06,
        "_disk_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("private")),
    )
    assert source_io_v06._decode_images([sample], state) == []
    assert state.exclusion_reasons[DDRExclusionReason.IMAGE_DECODE_ERROR.value] == 1

    state = DDRDiagnosticState()
    decoded = source_io_v06._DecodedImage(sample, np.zeros((4, 4, 3), dtype=np.uint8))
    monkeypatch.setattr(
        source_io_v06,
        "extract_ddr_retinal_fov",
        lambda *args: (_ for _ in ()).throw(ValueError("private")),
    )
    assert source_io_v06._extract_fovs([decoded], state) == []
    assert state.exclusion_reasons[DDRExclusionReason.FOV_EXTRACTION_ERROR.value] == 1

    state = DDRDiagnosticState()
    valid = source_io_v06._ValidatedFOV(sample, np.ones((4, 4), dtype=np.bool_))
    assert source_io_v06._decode_masks([valid], state) == []
    assert state.exclusion_reasons[DDRExclusionReason.MASK_DECODE_ERROR.value] == 1

    state = DDRDiagnosticState()
    decoded_masks = source_io_v06._DecodedMasks(
        sample,
        np.ones((4, 4), dtype=np.bool_),
        {axis: np.zeros((4, 4), dtype=np.bool_) for axis in ("ma", "hem", "ex", "cws")},
    )
    monkeypatch.setattr(
        source_io_v06, "_burden", lambda *args: (_ for _ in ()).throw(ValueError("private"))
    )
    assert source_io_v06._measure_burdens([decoded_masks], state) == []
    assert state.exclusion_reasons[DDRExclusionReason.BURDEN_MEASUREMENT_ERROR.value] == 1

    state = DDRDiagnosticState()
    measured = source_io_v06._MeasuredBurden(
        sample, {axis: 0.0 for axis in ("ma", "hem", "ex", "cws")}
    )
    monkeypatch.setattr(
        source_io_v06,
        "MeasurementRecord",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("private")),
    )
    assert source_io_v06._construct_records([measured], state) == []
    assert state.exclusion_reasons[DDRExclusionReason.RECORD_CONSTRUCTION_ERROR.value] == 1


def test_any_exclusion_fails_at_source_summary_with_aggregate_counts(tmp_path: Path) -> None:
    state = DDRDiagnosticState(observed_input_count=1)
    state.exclude(DDRExclusionReason.FOV_EMPTY)
    sample = _sample(tmp_path, identifier="must-not-appear")

    with pytest.raises(DDRSourceMeasurementError) as captured:
        source_io_v06._run_substage(
            state,
            DDRSubstage.SOURCE_SUMMARY,
            lambda: source_io_v06._source_summary([sample], [], state),
        )

    error = captured.value
    assert error.substage is DDRSubstage.SOURCE_SUMMARY
    assert error.error_family == "MEASUREMENT_EXCLUSION"
    assert error.diagnostics["measurement_exclusion_count"] == 1
    assert error.diagnostics["exclusion_reason_counts"] == {"DDR_FOV_EMPTY": 1}
    assert "must-not-appear" not in json.dumps(error.diagnostics)


def test_zero_exclusion_diagnostics_preserve_inherited_measurements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = (_sample(tmp_path, identifier="opaque-a"), _sample(tmp_path, identifier="opaque-b"))

    class FakeAdapter:
        def __init__(self, root: str | Path) -> None:
            self.root = root

        def validate(self) -> ValidationResult:
            return ValidationResult(AdapterStatus.READY)

        def samples(self) -> tuple[DatasetSample, ...]:
            return samples

    def image(path: Path, *, rgb: bool) -> np.ndarray:
        if rgb:
            return np.full((4, 4, 3), 20, dtype=np.uint8)
        return np.zeros((4, 4), dtype=np.uint8)

    for module in (source_io, source_io_v06):
        monkeypatch.setattr(module, "DDRAdapter", FakeAdapter)
        monkeypatch.setattr(module, "_disk_image", image)
        monkeypatch.setattr(module, "extract_ddr_retinal_fov", lambda *args: _valid_fov())

    inherited = source_io.load_ddr_confirmation_measurements(tmp_path)
    amended = source_io_v06.load_ddr_confirmation_measurements_v06(tmp_path)

    assert amended.measurements == inherited
    assert amended.diagnostics["measurement_eligible_count"] == 2
    assert amended.diagnostics["measurement_exclusion_count"] == 0
    assert amended.diagnostics["exclusion_reason_counts"] == {}
    assert amended.diagnostics["source_identifiers_recorded"] is False

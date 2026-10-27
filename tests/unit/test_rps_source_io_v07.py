from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nts_med.data.adapters import AdapterStatus, DatasetSample, ValidationResult
from rps_med import source_io_v06, source_io_v07
from rps_med.source_io_v07 import DDRDiagnosticStateV07, DDRFOVResolution


def _sample(tmp_path: Path, identifier: str, split: str) -> DatasetSample:
    return DatasetSample(
        sample_id=identifier,
        image_path=tmp_path / f"{identifier}.jpg",
        dataset_name="ddr",
        official_split=split,
        lesion_masks={
            axis: tmp_path / f"{identifier}-{axis}.png"
            for axis in ("ma", "hem", "ex", "cws")
        },
    )


def _single_component() -> np.ndarray:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[10:90, 10:90] = 80
    return image


def _dominant_multiple() -> np.ndarray:
    image = _single_component()
    image[0:5, 0:5] = 80
    return image


def test_v07_safe_diagnostics_add_only_fixed_aggregate_resolver_counts() -> None:
    state = DDRDiagnosticStateV07()
    state.fov_resolution_counts[DDRFOVResolution.DOMINANT_COMPONENT_ACCEPTED.value] = 1
    payload = state.safe_diagnostics()

    assert payload["dominant_component_min_share"] == 0.90
    assert payload["fov_resolution_counts"] == {
        "DDR_FOV_DOMINANT_COMPONENT_ACCEPTED": 1
    }
    assert payload["absolute_paths_recorded"] is False
    assert payload["source_identifiers_recorded"] is False


def test_development_audit_is_aggregate_only_and_preserves_legacy_valid_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = (
        _sample(tmp_path, "opaque-a", "train"),
        _sample(tmp_path, "opaque-b", "train"),
    )

    class FakeAdapter:
        def __init__(self, root: str | Path) -> None:
            self.root = root

        def validate(self) -> ValidationResult:
            return ValidationResult(AdapterStatus.READY)

        def samples(self) -> tuple[DatasetSample, ...]:
            return samples

    images = {
        samples[0].image_path: _single_component(),
        samples[1].image_path: _dominant_multiple(),
    }
    monkeypatch.setattr(source_io_v07, "DDRAdapter", FakeAdapter)
    monkeypatch.setattr(source_io_v07, "_disk_image", lambda path, **kwargs: images[path])

    payload = source_io_v07.audit_ddr_train_fov_v07(tmp_path)

    assert payload["observed_input_count"] == 2
    assert payload["legacy_valid_count"] == 1
    assert payload["legacy_valid_mask_equal_count"] == 1
    assert payload["legacy_valid_mask_difference_count"] == 0
    assert payload["candidate_newly_resolved_count"] == 1
    assert payload["source_identifiers_recorded"] is False
    assert "opaque-a" not in str(payload)


def test_confirmation_reader_uses_v07_resolver_without_case_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = _sample(tmp_path, "opaque-confirmation", "valid")

    class FakeAdapter:
        def __init__(self, root: str | Path) -> None:
            self.root = root

        def validate(self) -> ValidationResult:
            return ValidationResult(AdapterStatus.READY)

        def samples(self) -> tuple[DatasetSample, ...]:
            return (sample,)

    def image(path: Path, *, rgb: bool) -> np.ndarray:
        if rgb:
            return _dominant_multiple()
        return np.zeros((100, 100), dtype=np.uint8)

    monkeypatch.setattr(source_io_v06, "DDRAdapter", FakeAdapter)
    monkeypatch.setattr(source_io_v06, "_disk_image", image)

    outcome = source_io_v07.load_ddr_confirmation_measurements_v07(tmp_path)

    assert len(outcome.measurements.records) == 1
    assert outcome.measurements.measurement_exclusion_count == 0
    assert outcome.diagnostics["fov_resolution_counts"] == {
        "DDR_FOV_DOMINANT_COMPONENT_ACCEPTED": 1
    }
    assert outcome.diagnostics["source_identifiers_recorded"] is False

"""Privacy-safe DDR measurement diagnostics for formal RPS-Med R1 v0.6.

This technical amendment preserves the v0.5 measurement operations.  It only
separates their orchestration into fixed substages and aggregate exclusion
reason counts.  Paths, exception messages, source identifiers, and row-level
measurements are never included in diagnostics.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from nts_med.concepts.fov import FOVResult, FOVStatus, extract_ddr_retinal_fov
from nts_med.data.adapters import AdapterStatus, DatasetSample, DDRAdapter
from nts_med.gate0.execution import MeasurementRecord, SourceMeasurements
from nts_med.gate0.source_io import _binary_mask, _burden, _disk_image

T = TypeVar("T")


class DDRSubstage(StrEnum):
    """Fixed, privacy-safe DDR confirmation measurement substages."""

    SAMPLE_SCOPE = "DDR_SAMPLE_SCOPE"
    IMAGE_DECODE = "DDR_IMAGE_DECODE"
    FOV_EXTRACTION = "DDR_FOV_EXTRACTION"
    FOV_VALIDATION = "DDR_FOV_VALIDATION"
    MASK_DECODE = "DDR_MASK_DECODE"
    BURDEN_MEASUREMENT = "DDR_BURDEN_MEASUREMENT"
    RECORD_CONSTRUCTION = "DDR_RECORD_CONSTRUCTION"
    SOURCE_SUMMARY = "DDR_SOURCE_SUMMARY"


class DDRExclusionReason(StrEnum):
    """Fixed aggregate reason codes; none identify a source instance."""

    IMAGE_DECODE_ERROR = "DDR_IMAGE_DECODE_ERROR"
    FOV_EXTRACTION_ERROR = "DDR_FOV_EXTRACTION_ERROR"
    FOV_EMPTY = "DDR_FOV_EMPTY"
    FOV_MULTIPLE_COMPONENTS = "DDR_FOV_MULTIPLE_COMPONENTS"
    FOV_AREA_RATIO_OUT_OF_RANGE = "DDR_FOV_AREA_RATIO_OUT_OF_RANGE"
    FOV_OTHER_INVALID = "DDR_FOV_OTHER_INVALID"
    FOV_RATIO_NONFINITE = "DDR_FOV_RATIO_NONFINITE"
    MASK_DECODE_ERROR = "DDR_MASK_DECODE_ERROR"
    BURDEN_MEASUREMENT_ERROR = "DDR_BURDEN_MEASUREMENT_ERROR"
    RECORD_CONSTRUCTION_ERROR = "DDR_RECORD_CONSTRUCTION_ERROR"


_REASON_SUBSTAGE: Mapping[DDRExclusionReason, DDRSubstage] = {
    DDRExclusionReason.IMAGE_DECODE_ERROR: DDRSubstage.IMAGE_DECODE,
    DDRExclusionReason.FOV_EXTRACTION_ERROR: DDRSubstage.FOV_EXTRACTION,
    DDRExclusionReason.FOV_EMPTY: DDRSubstage.FOV_VALIDATION,
    DDRExclusionReason.FOV_MULTIPLE_COMPONENTS: DDRSubstage.FOV_VALIDATION,
    DDRExclusionReason.FOV_AREA_RATIO_OUT_OF_RANGE: DDRSubstage.FOV_VALIDATION,
    DDRExclusionReason.FOV_OTHER_INVALID: DDRSubstage.FOV_VALIDATION,
    DDRExclusionReason.FOV_RATIO_NONFINITE: DDRSubstage.FOV_VALIDATION,
    DDRExclusionReason.MASK_DECODE_ERROR: DDRSubstage.MASK_DECODE,
    DDRExclusionReason.BURDEN_MEASUREMENT_ERROR: DDRSubstage.BURDEN_MEASUREMENT,
    DDRExclusionReason.RECORD_CONSTRUCTION_ERROR: DDRSubstage.RECORD_CONSTRUCTION,
}


@dataclass(slots=True)
class DDRDiagnosticState:
    """Aggregate-only state deliberately excluding identifiers and paths."""

    current: DDRSubstage | None = None
    completed: list[DDRSubstage] = field(default_factory=list)
    observed_input_count: int | None = None
    eligible_count: int | None = None
    exclusion_count: int = 0
    exclusion_reasons: Counter[str] = field(default_factory=Counter)
    exclusion_substages: Counter[str] = field(default_factory=Counter)

    def enter(self, substage: DDRSubstage) -> None:
        self.current = substage

    def complete(self, substage: DDRSubstage) -> None:
        if self.current is not substage:
            raise RuntimeError("DDR substage completion is out of order")
        self.completed.append(substage)

    def exclude(self, reason: DDRExclusionReason) -> None:
        self.exclusion_count += 1
        self.exclusion_reasons[reason.value] += 1
        self.exclusion_substages[_REASON_SUBSTAGE[reason].value] += 1

    def safe_diagnostics(self) -> dict[str, object]:
        return {
            "completed_substages": [stage.value for stage in self.completed],
            "last_completed_substage": (self.completed[-1].value if self.completed else None),
            "observed_input_count": self.observed_input_count,
            "measurement_eligible_count": self.eligible_count,
            "measurement_exclusion_count": self.exclusion_count,
            "exclusion_reason_counts": dict(sorted(self.exclusion_reasons.items())),
            "exclusion_substage_counts": dict(sorted(self.exclusion_substages.items())),
            "exception_message_recorded": False,
            "traceback_recorded": False,
            "absolute_paths_recorded": False,
            "source_identifiers_recorded": False,
            "row_level_measurements_recorded": False,
        }


class DDRMeasurementExclusionError(ValueError):
    """Internal signal that the frozen zero-exclusion contract was violated."""


class DDRSourceMeasurementError(RuntimeError):
    """Sanitized technical abort carrying only fixed codes and counts."""

    def __init__(
        self,
        *,
        substage: DDRSubstage,
        error_family: str,
        diagnostics: Mapping[str, object],
    ) -> None:
        super().__init__("DDR_CONFIRMATION_MEASUREMENT_SUBSTAGE_ABORT")
        self.substage = substage
        self.error_family = error_family
        self.diagnostics = dict(diagnostics)


@dataclass(frozen=True, slots=True)
class DDRMeasurementOutcome:
    """In-memory measurements plus aggregate-only diagnostics."""

    measurements: SourceMeasurements
    diagnostics: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _DecodedImage:
    sample: DatasetSample
    image: NDArray[Any]


@dataclass(frozen=True, slots=True)
class _ExtractedFOV:
    sample: DatasetSample
    result: FOVResult


@dataclass(frozen=True, slots=True)
class _ValidatedFOV:
    sample: DatasetSample
    mask: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class _DecodedMasks:
    sample: DatasetSample
    fov: NDArray[np.bool_]
    masks: Mapping[str, NDArray[np.bool_]]


@dataclass(frozen=True, slots=True)
class _MeasuredBurden:
    sample: DatasetSample
    burden: Mapping[str, float]


def _ddr_error_family(error: Exception) -> str:
    if isinstance(error, DDRMeasurementExclusionError):
        return "MEASUREMENT_EXCLUSION"
    if isinstance(error, PermissionError):
        return "PERMISSION_ERROR"
    if isinstance(error, KeyError):
        return "KEY_ERROR"
    if isinstance(error, TypeError):
        return "TYPE_ERROR"
    if isinstance(error, ValueError):
        return "VALUE_ERROR"
    if isinstance(error, OSError):
        return "IO_ERROR"
    if isinstance(error, RuntimeError):
        return "RUNTIME_ERROR"
    return "UNEXPECTED_ERROR"


def _run_substage(
    state: DDRDiagnosticState,
    substage: DDRSubstage,
    operation: Callable[[], T],
) -> T:
    """Run one aggregate operation and discard exception detail on failure."""

    state.enter(substage)
    try:
        value = operation()
    except DDRSourceMeasurementError:
        raise
    except Exception as error:
        raise DDRSourceMeasurementError(
            substage=substage,
            error_family=_ddr_error_family(error),
            diagnostics=state.safe_diagnostics(),
        ) from None
    state.complete(substage)
    return value


def _valid_samples(root: str | Path, state: DDRDiagnosticState) -> tuple[DatasetSample, ...]:
    adapter = DDRAdapter(root)
    validation = adapter.validate()
    if validation.status is not AdapterStatus.READY:
        raise ValueError("DDR adapter must be ready")
    samples = tuple(sample for sample in adapter.samples() if sample.official_split == "valid")
    state.observed_input_count = len(samples)
    return samples


def _decode_images(
    samples: Sequence[DatasetSample], state: DDRDiagnosticState
) -> list[_DecodedImage]:
    decoded: list[_DecodedImage] = []
    for sample in samples:
        try:
            decoded.append(_DecodedImage(sample, _disk_image(sample.image_path, rgb=True)))
        except (KeyError, OSError, TypeError, ValueError):
            state.exclude(DDRExclusionReason.IMAGE_DECODE_ERROR)
    return decoded


def _extract_fovs(
    decoded: Sequence[_DecodedImage], state: DDRDiagnosticState
) -> list[_ExtractedFOV]:
    extracted: list[_ExtractedFOV] = []
    for item in decoded:
        try:
            extracted.append(_ExtractedFOV(item.sample, extract_ddr_retinal_fov(item.image)))
        except (KeyError, OSError, TypeError, ValueError):
            state.exclude(DDRExclusionReason.FOV_EXTRACTION_ERROR)
    return extracted


def _invalid_fov_reason(result: FOVResult) -> DDRExclusionReason | None:
    if result.status is FOVStatus.EMPTY:
        return DDRExclusionReason.FOV_EMPTY
    if result.status is FOVStatus.MULTIPLE_COMPONENTS:
        return DDRExclusionReason.FOV_MULTIPLE_COMPONENTS
    if result.status is FOVStatus.AREA_RATIO_OUT_OF_RANGE:
        return DDRExclusionReason.FOV_AREA_RATIO_OUT_OF_RANGE
    if result.status is not FOVStatus.VALID or result.mask is None:
        return DDRExclusionReason.FOV_OTHER_INVALID
    if result.area_ratio is None or not math.isfinite(result.area_ratio):
        return DDRExclusionReason.FOV_RATIO_NONFINITE
    return None


def _validate_fovs(
    extracted: Sequence[_ExtractedFOV], state: DDRDiagnosticState
) -> list[_ValidatedFOV]:
    valid: list[_ValidatedFOV] = []
    for item in extracted:
        reason = _invalid_fov_reason(item.result)
        if reason is not None:
            state.exclude(reason)
            continue
        assert item.result.mask is not None
        valid.append(_ValidatedFOV(item.sample, item.result.mask))
    return valid


def _decode_masks(valid: Sequence[_ValidatedFOV], state: DDRDiagnosticState) -> list[_DecodedMasks]:
    decoded: list[_DecodedMasks] = []
    for item in valid:
        try:
            masks = {
                axis: _binary_mask(
                    _disk_image(item.sample.lesion_masks[axis], rgb=False),
                    positive_class=255,
                )
                for axis in ("ma", "hem", "ex", "cws")
            }
            decoded.append(_DecodedMasks(item.sample, item.mask, masks))
        except (KeyError, OSError, TypeError, ValueError):
            state.exclude(DDRExclusionReason.MASK_DECODE_ERROR)
    return decoded


def _measure_burdens(
    decoded: Sequence[_DecodedMasks], state: DDRDiagnosticState
) -> list[_MeasuredBurden]:
    measured: list[_MeasuredBurden] = []
    for item in decoded:
        try:
            measured.append(_MeasuredBurden(item.sample, _burden(item.masks, item.fov)))
        except (KeyError, OSError, TypeError, ValueError):
            state.exclude(DDRExclusionReason.BURDEN_MEASUREMENT_ERROR)
    return measured


def _construct_records(
    measured: Sequence[_MeasuredBurden], state: DDRDiagnosticState
) -> list[MeasurementRecord]:
    records: list[MeasurementRecord] = []
    for item in measured:
        try:
            records.append(
                MeasurementRecord(
                    instance_id=item.sample.sample_id,
                    primary=item.burden,
                    variants={},
                    grade=None,
                    grading_split_conflict=False,
                    strict_measurement_quality=True,
                )
            )
        except (KeyError, OSError, TypeError, ValueError):
            state.exclude(DDRExclusionReason.RECORD_CONSTRUCTION_ERROR)
    return records


def _source_summary(
    samples: Sequence[DatasetSample],
    records: Sequence[MeasurementRecord],
    state: DDRDiagnosticState,
) -> SourceMeasurements:
    state.eligible_count = len(records)
    if len(records) + state.exclusion_count != len(samples):
        raise RuntimeError("DDR aggregate accounting is inconsistent")
    if state.exclusion_count:
        raise DDRMeasurementExclusionError("DDR zero-exclusion contract failed")
    return SourceMeasurements(
        name="ddr",
        role="FORMAL_R1_CONFIRMATION",
        expected_input_count=len(samples),
        observed_input_count=len(samples),
        records=tuple(records),
        measurement_exclusion_count=0,
    )


def load_ddr_confirmation_measurements_v06(root: str | Path) -> DDRMeasurementOutcome:
    """Measure DDR valid only after a future v0.6 authorization."""

    state = DDRDiagnosticState()
    samples = _run_substage(state, DDRSubstage.SAMPLE_SCOPE, lambda: _valid_samples(root, state))
    decoded_images = _run_substage(
        state, DDRSubstage.IMAGE_DECODE, lambda: _decode_images(samples, state)
    )
    extracted_fovs = _run_substage(
        state,
        DDRSubstage.FOV_EXTRACTION,
        lambda: _extract_fovs(decoded_images, state),
    )
    valid_fovs = _run_substage(
        state,
        DDRSubstage.FOV_VALIDATION,
        lambda: _validate_fovs(extracted_fovs, state),
    )
    decoded_masks = _run_substage(
        state, DDRSubstage.MASK_DECODE, lambda: _decode_masks(valid_fovs, state)
    )
    measured = _run_substage(
        state,
        DDRSubstage.BURDEN_MEASUREMENT,
        lambda: _measure_burdens(decoded_masks, state),
    )
    records = _run_substage(
        state,
        DDRSubstage.RECORD_CONSTRUCTION,
        lambda: _construct_records(measured, state),
    )
    measurements = _run_substage(
        state,
        DDRSubstage.SOURCE_SUMMARY,
        lambda: _source_summary(samples, records, state),
    )
    return DDRMeasurementOutcome(
        measurements=measurements,
        diagnostics=state.safe_diagnostics(),
    )


__all__ = [
    "DDRDiagnosticState",
    "DDRExclusionReason",
    "DDRMeasurementOutcome",
    "DDRSourceMeasurementError",
    "DDRSubstage",
    "load_ddr_confirmation_measurements_v06",
]

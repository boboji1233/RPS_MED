"""DDR FOV amendment and aggregate diagnostics for formal RPS-Med R1 v0.7."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np

from nts_med.concepts.fov import FOVResult, FOVStatus, extract_ddr_retinal_fov
from nts_med.data.adapters import AdapterStatus, DDRAdapter
from nts_med.gate0.source_io import _disk_image
from rps_med import source_io_v06 as v06
from rps_med.fov_v07 import (
    DDR_DOMINANT_COMPONENT_MIN_SHARE,
    DDR_FOV_V07_ALGORITHM_VERSION,
    extract_ddr_retinal_fov_v07,
)


class DDRFOVResolution(StrEnum):
    """Fixed aggregate-only v0.7 FOV resolution categories."""

    LEGACY_VALID = "DDR_FOV_LEGACY_VALID"
    LEGACY_EMPTY = "DDR_FOV_LEGACY_EMPTY"
    LEGACY_AREA_RATIO_OUT_OF_RANGE = "DDR_FOV_LEGACY_AREA_RATIO_OUT_OF_RANGE"
    DOMINANT_COMPONENT_ACCEPTED = "DDR_FOV_DOMINANT_COMPONENT_ACCEPTED"
    AMBIGUOUS_MULTIPLE_COMPONENTS = "DDR_FOV_AMBIGUOUS_MULTIPLE_COMPONENTS"


@dataclass(slots=True)
class DDRDiagnosticStateV07(v06.DDRDiagnosticState):
    """Extend v0.6 diagnostics with aggregate resolver counts only."""

    fov_resolution_counts: Counter[str] = field(default_factory=Counter)
    dominant_component_min_share: float = DDR_DOMINANT_COMPONENT_MIN_SHARE

    def safe_diagnostics(self) -> dict[str, object]:
        payload = v06.DDRDiagnosticState.safe_diagnostics(self)
        payload.update(
            {
                "fov_algorithm_version": DDR_FOV_V07_ALGORITHM_VERSION,
                "dominant_component_min_share": self.dominant_component_min_share,
                "fov_resolution_counts": dict(sorted(self.fov_resolution_counts.items())),
            }
        )
        return payload


def _resolution(result: FOVResult) -> DDRFOVResolution:
    if result.algorithm_version == DDR_FOV_V07_ALGORITHM_VERSION:
        if result.status is FOVStatus.VALID:
            return DDRFOVResolution.DOMINANT_COMPONENT_ACCEPTED
        return DDRFOVResolution.AMBIGUOUS_MULTIPLE_COMPONENTS
    if result.status is FOVStatus.VALID:
        return DDRFOVResolution.LEGACY_VALID
    if result.status is FOVStatus.EMPTY:
        return DDRFOVResolution.LEGACY_EMPTY
    if result.status is FOVStatus.AREA_RATIO_OUT_OF_RANGE:
        return DDRFOVResolution.LEGACY_AREA_RATIO_OUT_OF_RANGE
    return DDRFOVResolution.AMBIGUOUS_MULTIPLE_COMPONENTS


def _extract_fovs_v07(
    decoded: Sequence[v06._DecodedImage], state: DDRDiagnosticStateV07
) -> list[v06._ExtractedFOV]:
    extracted: list[v06._ExtractedFOV] = []
    for item in decoded:
        try:
            result = extract_ddr_retinal_fov_v07(
                item.image,
                dominant_component_min_share=state.dominant_component_min_share,
            )
            state.fov_resolution_counts[_resolution(result).value] += 1
            extracted.append(v06._ExtractedFOV(item.sample, result))
        except (KeyError, OSError, TypeError, ValueError):
            state.exclude(v06.DDRExclusionReason.FOV_EXTRACTION_ERROR)
    return extracted


def load_ddr_confirmation_measurements_v07(
    root: str | Path,
    *,
    dominant_component_min_share: float = DDR_DOMINANT_COMPONENT_MIN_SHARE,
) -> v06.DDRMeasurementOutcome:
    """Measure DDR valid under the separately authorized v0.7 FOV amendment."""

    state = DDRDiagnosticStateV07(
        dominant_component_min_share=dominant_component_min_share
    )
    samples = v06._run_substage(
        state, v06.DDRSubstage.SAMPLE_SCOPE, lambda: v06._valid_samples(root, state)
    )
    decoded_images = v06._run_substage(
        state, v06.DDRSubstage.IMAGE_DECODE, lambda: v06._decode_images(samples, state)
    )
    extracted_fovs = v06._run_substage(
        state,
        v06.DDRSubstage.FOV_EXTRACTION,
        lambda: _extract_fovs_v07(decoded_images, state),
    )
    valid_fovs = v06._run_substage(
        state,
        v06.DDRSubstage.FOV_VALIDATION,
        lambda: v06._validate_fovs(extracted_fovs, state),
    )
    decoded_masks = v06._run_substage(
        state, v06.DDRSubstage.MASK_DECODE, lambda: v06._decode_masks(valid_fovs, state)
    )
    measured = v06._run_substage(
        state,
        v06.DDRSubstage.BURDEN_MEASUREMENT,
        lambda: v06._measure_burdens(decoded_masks, state),
    )
    records = v06._run_substage(
        state,
        v06.DDRSubstage.RECORD_CONSTRUCTION,
        lambda: v06._construct_records(measured, state),
    )
    measurements = v06._run_substage(
        state,
        v06.DDRSubstage.SOURCE_SUMMARY,
        lambda: v06._source_summary(samples, records, state),
    )
    return v06.DDRMeasurementOutcome(
        measurements=measurements,
        diagnostics=state.safe_diagnostics(),
    )


def audit_ddr_train_fov_v07(
    root: str | Path,
    *,
    dominant_component_min_share: float = DDR_DOMINANT_COMPONENT_MIN_SHARE,
) -> Mapping[str, object]:
    """Compare legacy and v0.7 FOV masks on DDR official train, aggregate-only."""

    adapter = DDRAdapter(root)
    validation = adapter.validate()
    if validation.status is not AdapterStatus.READY:
        raise ValueError("DDR adapter must be ready")
    samples = tuple(sample for sample in adapter.samples() if sample.official_split == "train")
    legacy_status: Counter[str] = Counter()
    candidate_status: Counter[str] = Counter()
    candidate_resolution: Counter[str] = Counter()
    decode_exclusions = 0
    legacy_valid = 0
    mask_equal = 0
    mask_different = 0
    newly_resolved = 0
    for sample in samples:
        try:
            image = _disk_image(sample.image_path, rgb=True)
            legacy = extract_ddr_retinal_fov(image)
            candidate = extract_ddr_retinal_fov_v07(
                image,
                dominant_component_min_share=dominant_component_min_share,
            )
        except (KeyError, OSError, TypeError, ValueError):
            decode_exclusions += 1
            continue
        legacy_status[legacy.status.value] += 1
        candidate_status[candidate.status.value] += 1
        candidate_resolution[_resolution(candidate).value] += 1
        if legacy.status is FOVStatus.VALID:
            legacy_valid += 1
            if (
                legacy.mask is not None
                and candidate.mask is not None
                and np.array_equal(legacy.mask, candidate.mask)
            ):
                mask_equal += 1
            else:
                mask_different += 1
        if (
            legacy.status is FOVStatus.MULTIPLE_COMPONENTS
            and candidate.status is FOVStatus.VALID
        ):
            newly_resolved += 1
    observed = len(samples)
    if sum(legacy_status.values()) + decode_exclusions != observed:
        raise RuntimeError("DDR train legacy FOV accounting is inconsistent")
    if sum(candidate_status.values()) + decode_exclusions != observed:
        raise RuntimeError("DDR train candidate FOV accounting is inconsistent")
    return {
        "scope": "DDR_LESION_OFFICIAL_TRAIN",
        "observed_input_count": observed,
        "image_decode_exclusion_count": decode_exclusions,
        "legacy_status_counts": dict(sorted(legacy_status.items())),
        "candidate_status_counts": dict(sorted(candidate_status.items())),
        "candidate_resolution_counts": dict(sorted(candidate_resolution.items())),
        "legacy_valid_count": legacy_valid,
        "legacy_valid_mask_equal_count": mask_equal,
        "legacy_valid_mask_difference_count": mask_different,
        "candidate_newly_resolved_count": newly_resolved,
        "dominant_component_min_share": dominant_component_min_share,
        "fov_algorithm_version": DDR_FOV_V07_ALGORITHM_VERSION,
        "lesion_masks_opened": False,
        "relations_computed": False,
        "local_paths_recorded": False,
        "source_identifiers_recorded": False,
        "row_level_measurements_recorded": False,
    }


__all__ = [
    "DDRDiagnosticStateV07",
    "DDRFOVResolution",
    "audit_ddr_train_fov_v07",
    "load_ddr_confirmation_measurements_v07",
]

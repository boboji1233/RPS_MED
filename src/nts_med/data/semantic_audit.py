"""Path-safe semantic audits for adapter-normalized lesion samples."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from nts_med.concepts.burden import DR_LESION_AXES
from nts_med.data.adapters.base import DatasetSample


class AuditStageStatus(StrEnum):
    """Completion state for one Prompt 2 audit stage."""

    COMPLETE = "COMPLETE"
    ISSUES_FOUND = "ISSUES_FOUND"
    NOT_RUN = "NOT_RUN"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class AxisMaskAudit:
    """Aggregate mask facts that contain no source identifier or path."""

    axis: str
    expected_instances: int
    masks_present: int
    positive_instances: int
    empty_instances: int
    unreadable_masks: int
    invalid_value_masks: int
    size_mismatch_masks: int

    @property
    def matching_rate(self) -> float | None:
        if self.expected_instances == 0:
            return None
        return self.masks_present / self.expected_instances

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["matching_rate"] = self.matching_rate
        return value


@dataclass(frozen=True, slots=True)
class SemanticAudit:
    """Aggregate semantic audit for samples emitted by a ready adapter."""

    status: AuditStageStatus
    sample_count: int
    official_split_counts: Mapping[str, int]
    grade_distributions: Mapping[str, Mapping[str, int]]
    missing_patient_id: int
    missing_eye_id: int
    missing_quality_fields: int
    unreadable_images: int
    image_channel_counts: Mapping[str, int]
    image_size_counts: Mapping[str, int]
    axes: tuple[AxisMaskAudit, ...]
    issue_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "audit_scope": "LOCAL_FILE_AND_MASK_SCAN_ONLY",
            "scientific_semantics_qualified": False,
            "sample_count": self.sample_count,
            "official_split_counts": dict(self.official_split_counts),
            "grade_distributions": {
                field: dict(distribution)
                for field, distribution in self.grade_distributions.items()
            },
            "missing_patient_id": self.missing_patient_id,
            "missing_eye_id": self.missing_eye_id,
            "missing_quality_fields": self.missing_quality_fields,
            "unreadable_images": self.unreadable_images,
            "image_channel_counts": dict(self.image_channel_counts),
            "image_size_counts": dict(self.image_size_counts),
            "axes": [axis.to_dict() for axis in self.axes],
            "issue_counts": dict(self.issue_counts),
        }


@dataclass(slots=True)
class _MutableAxisAudit:
    masks_present: int = 0
    positive_instances: int = 0
    empty_instances: int = 0
    unreadable_masks: int = 0
    invalid_value_masks: int = 0
    size_mismatch_masks: int = 0


def _safe_grade(value: object) -> str:
    if value is None:
        return "MISSING"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, int)):
        text = str(value)
        return text if text else "MISSING"
    if isinstance(value, float) and math.isfinite(value):
        return str(value)
    return "INVALID"


def _read_image_facts(path: Path) -> tuple[tuple[int, int], int]:
    with Image.open(path) as image:
        image.load()
        size = (int(image.width), int(image.height))
        channels = len(image.getbands())
    return size, channels


def _read_mask(path: Path) -> tuple[np.ndarray[Any, Any], tuple[int, int]]:
    with Image.open(path) as image:
        image.load()
        if len(image.getbands()) != 1:
            raise ValueError("mask is not single-channel")
        array = np.asarray(image)
        size = (int(image.width), int(image.height))
    if array.ndim != 2:
        raise ValueError("mask is not two-dimensional")
    return array, size


def audit_samples(
    samples: Iterable[DatasetSample],
    *,
    axes: Iterable[str] = DR_LESION_AXES,
    positive_class: int = 255,
) -> SemanticAudit:
    """Audit normalized image/mask samples without returning paths or source IDs."""

    selected_axes = tuple(axes)
    if not selected_axes or len(set(selected_axes)) != len(selected_axes):
        raise ValueError("axes must be a non-empty unique sequence")
    if isinstance(positive_class, bool) or not isinstance(positive_class, int):
        raise TypeError("positive_class must be an integer")

    materialized = sorted(samples, key=lambda sample: sample.sample_id)
    identifiers = [sample.sample_id for sample in materialized]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("adapter emitted duplicate opaque sample identifiers")

    split_counts: Counter[str] = Counter()
    grade_counts: dict[str, Counter[str]] = {}
    image_channels: Counter[str] = Counter()
    image_sizes: Counter[str] = Counter()
    issues: Counter[str] = Counter()
    axis_values = {axis: _MutableAxisAudit() for axis in selected_axes}
    missing_patient = 0
    missing_eye = 0
    missing_quality = 0
    unreadable_images = 0

    for sample in materialized:
        split = getattr(sample, "official_split", None)
        split_counts[split if isinstance(split, str) and split else "UNKNOWN"] += 1
        missing_patient += sample.patient_id is None
        missing_eye += sample.eye_id is None
        missing_quality += not bool(sample.quality_fields)
        for field, value in sample.grade_fields.items():
            grade_counts.setdefault(field, Counter())[_safe_grade(value)] += 1

        image_size: tuple[int, int] | None = None
        try:
            image_size, channels = _read_image_facts(sample.image_path)
            image_channels[str(channels)] += 1
            image_sizes[f"{image_size[0]}x{image_size[1]}"] += 1
        except (OSError, UnidentifiedImageError, ValueError, SyntaxError):
            unreadable_images += 1
            issues["UNREADABLE_IMAGE"] += 1

        for axis in selected_axes:
            aggregate = axis_values[axis]
            mask_path = sample.lesion_masks.get(axis)
            if mask_path is None:
                issues[f"MISSING_MASK_{axis.upper()}"] += 1
                continue
            aggregate.masks_present += 1
            try:
                mask, mask_size = _read_mask(mask_path)
            except (OSError, UnidentifiedImageError, ValueError, SyntaxError):
                aggregate.unreadable_masks += 1
                issues[f"UNREADABLE_MASK_{axis.upper()}"] += 1
                continue
            values = np.unique(mask)
            if not bool(np.isin(values, np.asarray([0, positive_class])).all()):
                aggregate.invalid_value_masks += 1
                issues[f"INVALID_MASK_VALUES_{axis.upper()}"] += 1
                continue
            if image_size is not None and mask_size != image_size:
                aggregate.size_mismatch_masks += 1
                issues[f"MASK_SIZE_MISMATCH_{axis.upper()}"] += 1
            if bool(np.any(mask == positive_class)):
                aggregate.positive_instances += 1
            else:
                aggregate.empty_instances += 1

    axis_audits = tuple(
        AxisMaskAudit(
            axis=axis,
            expected_instances=len(materialized),
            masks_present=value.masks_present,
            positive_instances=value.positive_instances,
            empty_instances=value.empty_instances,
            unreadable_masks=value.unreadable_masks,
            invalid_value_masks=value.invalid_value_masks,
            size_mismatch_masks=value.size_mismatch_masks,
        )
        for axis, value in axis_values.items()
    )
    return SemanticAudit(
        status=(
            AuditStageStatus.ISSUES_FOUND if issues else AuditStageStatus.COMPLETE
        ),
        sample_count=len(materialized),
        official_split_counts=dict(sorted(split_counts.items())),
        grade_distributions={
            field: dict(sorted(distribution.items()))
            for field, distribution in sorted(grade_counts.items())
        },
        missing_patient_id=missing_patient,
        missing_eye_id=missing_eye,
        missing_quality_fields=missing_quality,
        unreadable_images=unreadable_images,
        image_channel_counts=dict(sorted(image_channels.items())),
        image_size_counts=dict(sorted(image_sizes.items())),
        axes=axis_audits,
        issue_counts=dict(sorted(issues.items())),
    )


__all__ = [
    "AuditStageStatus",
    "AxisMaskAudit",
    "SemanticAudit",
    "audit_samples",
]

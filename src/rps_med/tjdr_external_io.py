"""Authorized-only TJDR official-test measurement boundary.

Callers must consume a frozen one-run authorization before invoking this
module.  Source paths and member names never appear in returned values.
"""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from nts_med.concepts.fov import FOVStatus, lesion_outside_fov_ratio
from nts_med.data.adapters import opaque_sample_id
from nts_med.gate0.execution import MeasurementRecord, SourceMeasurements
from nts_med.gate0.source_io import _burden
from rps_med.tjdr_io import (
    TJDR_AXES_TO_CLASSES,
    _index_split,
    _read_image,
    _validate_annotation,
    extract_tjdr_retinal_fov,
)


def load_tjdr_test_measurements(
    archive_paths: Sequence[str | Path],
    *,
    expected_count: int,
    background_threshold: float = 5.0,
    minimum_area_ratio: float = 0.10,
    maximum_area_ratio: float = 0.98,
    substantial_component_ratio: float = 0.001,
    dominant_component_min_share: float = 0.90,
    allow_all_background: bool = True,
) -> SourceMeasurements:
    """Measure official test after authorization; invalid cases fail as exclusions."""

    paths = tuple(Path(path) for path in archive_paths)
    records: list[MeasurementRecord] = []
    exclusions = 0
    with ExitStack() as stack:
        archives = tuple(stack.enter_context(zipfile.ZipFile(path)) for path in paths)
        pairs = _index_split(archives, split="test")
        for pair in pairs:
            try:
                image, _ = _read_image(archives, pair.image, rgb=True)
                annotation, _ = _read_image(archives, pair.annotation, rgb=False)
                _validate_annotation(
                    annotation,
                    image.shape[:2],
                    allow_all_background=allow_all_background,
                )
                result = extract_tjdr_retinal_fov(
                    image,
                    background_threshold=background_threshold,
                    minimum_area_ratio=minimum_area_ratio,
                    maximum_area_ratio=maximum_area_ratio,
                    substantial_component_ratio=substantial_component_ratio,
                    dominant_component_min_share=dominant_component_min_share,
                )
                if result.status is not FOVStatus.VALID or result.mask is None:
                    raise ValueError("TJDR test FOV is invalid")
                masks = {
                    axis: np.asarray(annotation == source_class, dtype=np.bool_)
                    for axis, source_class in TJDR_AXES_TO_CLASSES.items()
                }
                if any(lesion_outside_fov_ratio(mask, result.mask) > 0 for mask in masks.values()):
                    raise ValueError("TJDR test lesion lies outside the FOV")
                records.append(
                    MeasurementRecord(
                        instance_id=opaque_sample_id("tjdr", f"test/{pair.stem}"),
                        primary=_burden(masks, result.mask),
                        variants={},
                        grade=None,
                        grading_split_conflict=False,
                        strict_measurement_quality=True,
                    )
                )
            except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                exclusions += 1
    return SourceMeasurements(
        name="tjdr_test",
        role="INDEPENDENT_EXTERNAL_REPLICATION",
        expected_input_count=expected_count,
        observed_input_count=len(pairs),
        records=tuple(records),
        measurement_exclusion_count=exclusions,
    )


__all__ = ["load_tjdr_test_measurements"]

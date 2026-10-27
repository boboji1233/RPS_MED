"""Read-only source measurement for the formal Gate 0 v0.6 executor.

All identifiers remain opaque and all measurements stay in memory. Callers
must complete one-run authorization consumption before invoking this module.
"""

from __future__ import annotations

import csv
import io
import math
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from nts_med.concepts.burden import (
    DR_LESION_AXES,
    component_count_density,
    compute_dr_burden,
    compute_lesion_metrics,
    safe_log1p,
)
from nts_med.concepts.fov import (
    FOVStatus,
    binary_dilate,
    binary_erode,
    extract_ddr_bounding_ellipse_fov,
    extract_ddr_retinal_fov,
    official_elliptical_roi_fov,
)
from nts_med.data.adapters import AdapterStatus, DDRAdapter, opaque_sample_id
from nts_med.gate0.execution import (
    MeasurementRecord,
    ReaderMeasurements,
    SourceMeasurements,
)
from nts_med.gate0.methods import strict_measurement_quality_eligible
from nts_med.relations.algebra import Interval

_AXIS_DIRECTORIES: Mapping[str, str] = {
    "ma": "Microaneurysms",
    "hem": "Hemorrhages",
    "ex": "Exudates",
    "cws": "CottonWoolSpots",
}
_MINIMUMS = {axis: 1 for axis in DR_LESION_AXES}
_DENSITY_SCALE = 1_000_000.0


def _zip_image(archive: zipfile.ZipFile, member: str) -> NDArray[Any]:
    with archive.open(member) as stream, Image.open(stream) as image:
        image.load()
        return np.asarray(image).copy()


def _disk_image(path: Path, *, rgb: bool) -> NDArray[Any]:
    with Image.open(path) as image:
        image.load()
        converted = image.convert("RGB") if rgb else image
        return np.asarray(converted).copy()


def _binary_mask(mask: NDArray[Any], *, positive_class: int) -> NDArray[np.bool_]:
    if mask.ndim != 2 or any(size == 0 for size in mask.shape):
        raise ValueError("lesion mask must be a non-empty two-dimensional array")
    if not bool(np.logical_or(mask == 0, mask == positive_class).all()):
        raise ValueError("lesion mask encoding is invalid")
    return np.asarray(mask == positive_class, dtype=np.bool_)


def _custom_burden(
    masks: Mapping[str, NDArray[np.bool_]],
    fov: NDArray[np.bool_],
    modes: Mapping[str, str],
    *,
    minimum: int = 1,
) -> dict[str, float]:
    if set(masks) != set(DR_LESION_AXES) or set(modes) != set(DR_LESION_AXES):
        raise ValueError("custom burden axes must match the four-axis contract")
    burdens: dict[str, float] = {}
    for axis in DR_LESION_AXES:
        metrics = compute_lesion_metrics(
            masks[axis],
            fov,
            min_component_pixels=minimum,
            connectivity=8,
        )
        mode = modes[axis]
        if mode == "COMPONENT_COUNT":
            raw = component_count_density(metrics.component_count, metrics.fov_area)
        elif mode == "AREA":
            raw = metrics.lesion_area_ratio
        else:
            raise ValueError("unknown axis burden mode")
        burdens[axis] = safe_log1p(raw, scale=_DENSITY_SCALE)
    return burdens


def _burden(
    masks: Mapping[str, NDArray[np.bool_]],
    fov: NDArray[np.bool_],
    *,
    minimum: int = 1,
) -> dict[str, float]:
    return compute_dr_burden(
        masks,
        fov,
        {axis: minimum for axis in DR_LESION_AXES},
        scale=_DENSITY_SCALE,
        connectivity=8,
    )


def _variant_burdens(
    masks: Mapping[str, NDArray[np.bool_]],
    fov: NDArray[np.bool_],
    *,
    fov_ratio: float,
    fov_bounds: tuple[float, float],
    alternative_fov: NDArray[np.bool_] | None = None,
) -> tuple[dict[str, float], dict[str, Mapping[str, float] | None]]:
    primary = _burden(masks, fov)
    eroded_masks = {
        axis: binary_erode(mask, radius=1) for axis, mask in masks.items()
    }
    dilated_masks = {
        axis: binary_dilate(mask, radius=1) for axis, mask in masks.items()
    }
    eroded_fov = binary_erode(fov, radius=1)
    dilated_fov = binary_dilate(fov, radius=1)
    variants: dict[str, Mapping[str, float] | None] = {
        "MASK_EROSION_RADIUS1": _burden(eroded_masks, fov),
        "MASK_DILATION_RADIUS1": _burden(dilated_masks, fov),
        "FOV_EROSION_RADIUS1": (
            _burden(masks, eroded_fov) if bool(eroded_fov.any()) else None
        ),
        "FOV_DILATION_RADIUS1": (
            _burden(masks, dilated_fov) if bool(dilated_fov.any()) else None
        ),
        "FOV_EXTREME_Q01_Q99_EXCLUSION": (
            primary if fov_bounds[0] <= fov_ratio <= fov_bounds[1] else None
        ),
        "COMPONENT_THRESHOLD_0": _burden(masks, fov, minimum=0),
        "COMPONENT_THRESHOLD_2": _burden(masks, fov, minimum=2),
        "COMPONENT_THRESHOLD_4": _burden(masks, fov, minimum=4),
    }
    if alternative_fov is not None:
        variants["FOV_ALTERNATIVE_BOUNDING_ELLIPSE"] = _burden(
            masks, alternative_fov
        )
    primary_modes = {
        "ma": "COMPONENT_COUNT",
        "hem": "AREA",
        "ex": "AREA",
        "cws": "AREA",
    }
    for axis in DR_LESION_AXES:
        changed = dict(primary_modes)
        changed[axis] = "AREA" if axis == "ma" else "COMPONENT_COUNT"
        variants[f"AXIS_MEASURE_{axis.upper()}_{changed[axis]}"] = _custom_burden(
            masks, fov, changed
        )
    return primary, variants


def _roi_rows(archive: zipfile.ZipFile) -> dict[str, dict[str, int]]:
    with (
        archive.open("MESSIDOR-ROIs.csv") as binary,
        io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as stream,
    ):
        reader = csv.DictReader(stream)
        required = {"name", "x0", "y0", "x1", "y1", "H", "W"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("official MAPLES ROI table schema is invalid")
        rows: dict[str, dict[str, int]] = {}
        for row in reader:
            key = row["name"].strip().casefold()
            if not key or key in rows:
                raise ValueError("official MAPLES ROI keys must be unique")
            rows[key] = {field: int(row[field]) for field in required - {"name"}}
    return rows


def _maples_keys(archive: zipfile.ZipFile, split: str) -> tuple[str, ...]:
    prefix = f"{split}/Microaneurysms/"
    keys = tuple(
        sorted(
            PurePosixPath(member).stem.casefold()
            for member in archive.namelist()
            if member.startswith(prefix) and member.casefold().endswith(".png")
        )
    )
    if len(set(keys)) != len(keys):
        raise ValueError("MAPLES split keys must be unique")
    return keys


def _maples_grades(archive: zipfile.ZipFile, split: str) -> dict[str, int]:
    with (
        archive.open(f"{split}/diagnosis.csv") as binary,
        io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as stream,
    ):
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("MAPLES diagnosis table is missing a header")
        normalized = {field.casefold(): field for field in reader.fieldnames}
        name_field = normalized.get("name")
        grade_field = normalized.get("dr")
        if name_field is None or grade_field is None:
            raise ValueError("MAPLES diagnosis table schema is invalid")
        grades: dict[str, int] = {}
        for row in reader:
            key = PurePosixPath(row[name_field].strip()).stem.casefold()
            grade = int(row[grade_field])
            if not key or key in grades or not 0 <= grade <= 4:
                raise ValueError("MAPLES diagnosis row is invalid")
            grades[key] = grade
    return grades


def load_maples_test_measurements(
    main_archive_path: str | Path,
    additional_archive_path: str | Path,
    variability_archive_path: str | Path,
    *,
    fov_bounds: tuple[float, float],
) -> SourceMeasurements:
    """Open the reserved MAPLES test scope after authorization consumption."""

    with zipfile.ZipFile(main_archive_path) as main, zipfile.ZipFile(
        additional_archive_path
    ) as additional, zipfile.ZipFile(variability_archive_path) as variability:
        keys = _maples_keys(main, "test")
        if len(keys) != 60:
            raise ValueError("MAPLES official test count must be 60")
        rows = _roi_rows(additional)
        grades = _maples_grades(main, "test")
        member_index = {member.casefold(): member for member in main.namelist()}
        records: list[MeasurementRecord] = []
        key_to_id: dict[str, str] = {}
        exclusions = 0
        for key in keys:
            try:
                roi = rows[key]
                fov = official_elliptical_roi_fov(
                    (roi["H"], roi["W"]),
                    x0=roi["x0"],
                    y0=roi["y0"],
                    x1=roi["x1"],
                    y1=roi["y1"],
                )
                masks = {
                    axis: _binary_mask(
                        _zip_image(
                            main,
                            member_index[f"test/{directory}/{key}.png".casefold()],
                        ),
                        positive_class=1,
                    )
                    for axis, directory in _AXIS_DIRECTORIES.items()
                }
                ratio = float(np.mean(fov))
                primary, variants = _variant_burdens(
                    masks, fov, fov_ratio=ratio, fov_bounds=fov_bounds
                )
                identifier = opaque_sample_id(
                    "maples_dr", f"annotations/test/{key}"
                )
                key_to_id[key] = identifier
                records.append(
                    MeasurementRecord(
                        instance_id=identifier,
                        primary=primary,
                        variants=variants,
                        grade=grades.get(key),
                        grading_split_conflict=False,
                        strict_measurement_quality=True,
                    )
                )
            except (KeyError, OSError, TypeError, ValueError):
                exclusions += 1

        variability_index = {
            member.casefold(): member for member in variability.namelist()
        }
        reader_keys = tuple(
            sorted(
                key
                for key in keys
                if all(
                    f"manual/{reader}/{directory}/{key}.png".casefold()
                    in variability_index
                    for reader in ("S1", "S2", "S3")
                    for directory in _AXIS_DIRECTORIES.values()
                )
            )
        )
        if len(reader_keys) != 16:
            raise ValueError("MAPLES reserved complete reader count must be 16")
        reader_fov = official_elliptical_roi_fov(
            (1500, 1500), x0=0, y0=0, x1=1500, y1=1500
        )
        point_variants: dict[str, dict[str, Mapping[str, float]]] = {
            reader: {} for reader in ("S1", "S2", "S3")
        }
        envelopes: dict[str, dict[str, Interval]] = {}
        for key in reader_keys:
            reader_identifier = key_to_id.get(key)
            if reader_identifier is None:
                raise ValueError("reader instance is not primary eligible")
            by_reader: dict[str, dict[str, float]] = {}
            for reader in ("S1", "S2", "S3"):
                masks = {
                    axis: _binary_mask(
                        _zip_image(
                            variability,
                            variability_index[
                                f"manual/{reader}/{directory}/{key}.png".casefold()
                            ],
                        ),
                        positive_class=255,
                    )
                    for axis, directory in _AXIS_DIRECTORIES.items()
                }
                burden = _burden(masks, reader_fov)
                by_reader[reader] = burden
                point_variants[reader][reader_identifier] = burden
            envelopes[reader_identifier] = {
                axis: Interval(
                    min(by_reader[reader][axis] for reader in by_reader),
                    max(by_reader[reader][axis] for reader in by_reader),
                )
                for axis in DR_LESION_AXES
            }
    return SourceMeasurements(
        name="maples_dr_messidor1",
        role="CONFIRMATORY",
        expected_input_count=60,
        observed_input_count=len(keys),
        records=tuple(records),
        measurement_exclusion_count=exclusions,
        reader=ReaderMeasurements(point_variants=point_variants, envelopes=envelopes),
    )


def load_ddr_test_measurements(
    root: str | Path,
    *,
    fov_bounds: tuple[float, float],
) -> SourceMeasurements:
    """Open the reserved DDR lesion-test scope after authorization consumption."""

    adapter = DDRAdapter(root)
    validation = adapter.validate()
    if validation.status is not AdapterStatus.READY:
        raise ValueError("DDR adapter must be ready")
    samples = tuple(
        sample for sample in adapter.samples() if sample.official_split == "test"
    )
    if len(samples) != 225:
        raise ValueError("DDR lesion test count must be 225")
    records: list[MeasurementRecord] = []
    exclusions = 0
    for sample in samples:
        try:
            image = _disk_image(sample.image_path, rgb=True)
            fov_result = extract_ddr_retinal_fov(image)
            if fov_result.status is not FOVStatus.VALID or fov_result.mask is None:
                raise ValueError("DDR primary FOV is invalid")
            fov = fov_result.mask
            masks = {
                axis: _binary_mask(
                    _disk_image(sample.lesion_masks[axis], rgb=False),
                    positive_class=255,
                )
                for axis in DR_LESION_AXES
            }
            alternative = extract_ddr_bounding_ellipse_fov(image)
            alternative_fov = (
                alternative.mask
                if alternative.status is FOVStatus.VALID
                and alternative.mask is not None
                else None
            )
            if fov_result.area_ratio is None or not math.isfinite(fov_result.area_ratio):
                raise ValueError("DDR FOV ratio is unavailable")
            primary, variants = _variant_burdens(
                masks,
                fov,
                fov_ratio=fov_result.area_ratio,
                fov_bounds=fov_bounds,
                alternative_fov=alternative_fov,
            )
            low_quality = bool(sample.quality_fields.get("ungradable_grade5", False))
            grade_value = sample.grade_fields.get("dr_grade")
            grade = int(grade_value) if isinstance(grade_value, int) else None
            conflict = bool(sample.metadata.get("grading_split_conflict", False))
            strict = strict_measurement_quality_eligible(
                dataset="ddr",
                primary_eligible=True,
                source_low_quality=low_quality,
                fov_method=fov_result.method,
                border_contact=fov_result.border_contact,
            )
            records.append(
                MeasurementRecord(
                    instance_id=sample.sample_id,
                    primary=primary,
                    variants=variants,
                    grade=grade,
                    grading_split_conflict=conflict,
                    strict_measurement_quality=strict,
                )
            )
        except (KeyError, OSError, TypeError, ValueError):
            exclusions += 1
    return SourceMeasurements(
        name="ddr",
        role="INDEPENDENT_REPLICATION",
        expected_input_count=225,
        observed_input_count=len(samples),
        records=tuple(records),
        measurement_exclusion_count=exclusions,
    )


__all__ = ["load_ddr_test_measurements", "load_maples_test_measurements"]

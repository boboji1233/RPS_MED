"""Read-only development source boundaries and measurements for RPS-Med."""

from __future__ import annotations

import csv
import hashlib
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

from nts_med.concepts.fov import (
    FOVStatus,
    extract_ddr_retinal_fov,
    official_elliptical_roi_fov,
)
from nts_med.data.adapters import AdapterStatus, DDRAdapter, opaque_sample_id
from nts_med.gate0.execution import MeasurementRecord, SourceMeasurements
from nts_med.gate0.source_io import (
    _AXIS_DIRECTORIES,
    _binary_mask,
    _burden,
    _disk_image,
    _maples_keys,
    _roi_rows,
    _zip_image,
)
from nts_med.reproducibility.freeze import sha256_file


def _identifier_digest(values: tuple[str, ...]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class DevelopmentBoundary:
    """Identifier-free binding of development and sealed confirmation scopes."""

    maples_total_train: int
    maples_development_count: int
    maples_confirmation_count: int
    maples_development_digest: str
    maples_confirmation_digest: str
    ddr_development_count: int
    ddr_confirmation_count: int
    ddr_development_digest: str
    ddr_confirmation_digest: str


def _maples_partition(
    archive: zipfile.ZipFile, *, development_count: int
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    keys = _maples_keys(archive, "train")
    keyed = tuple(
        sorted(
            (
                opaque_sample_id("maples_dr", f"annotations/train/{key}"),
                key,
            )
            for key in keys
        )
    )
    if development_count <= 1 or development_count >= len(keyed) - 1:
        raise ValueError("MAPLES development count leaves an invalid boundary")
    development = keyed[:development_count]
    confirmation = keyed[development_count:]
    return (
        tuple(key for _, key in development),
        tuple(key for _, key in confirmation),
        tuple(identifier for identifier, _ in development),
        tuple(identifier for identifier, _ in confirmation),
    )


def inspect_development_boundary(
    maples_archive_path: str | Path,
    ddr_root: str | Path,
    *,
    maples_development_count: int,
) -> DevelopmentBoundary:
    """Inspect split structure without opening images or masks."""

    with zipfile.ZipFile(maples_archive_path) as archive:
        development, confirmation, development_ids, confirmation_ids = _maples_partition(
            archive, development_count=maples_development_count
        )
    adapter = DDRAdapter(ddr_root)
    validation = adapter.validate()
    if validation.status is not AdapterStatus.READY:
        raise ValueError("DDR adapter must be ready")
    samples = adapter.samples()
    train_ids = tuple(
        sorted(
            sample.sample_id for sample in samples if sample.official_split == "train"
        )
    )
    valid_ids = tuple(
        sorted(
            sample.sample_id for sample in samples if sample.official_split == "valid"
        )
    )
    return DevelopmentBoundary(
        maples_total_train=len(development) + len(confirmation),
        maples_development_count=len(development),
        maples_confirmation_count=len(confirmation),
        maples_development_digest=_identifier_digest(development_ids),
        maples_confirmation_digest=_identifier_digest(confirmation_ids),
        ddr_development_count=len(train_ids),
        ddr_confirmation_count=len(valid_ids),
        ddr_development_digest=_identifier_digest(train_ids),
        ddr_confirmation_digest=_identifier_digest(valid_ids),
    )


def load_maples_development_measurements(
    main_archive_path: str | Path,
    additional_archive_path: str | Path,
    *,
    development_count: int,
) -> SourceMeasurements:
    """Measure only the precommitted MAPLES train development partition."""

    with zipfile.ZipFile(main_archive_path) as main, zipfile.ZipFile(
        additional_archive_path
    ) as additional:
        selected, _, selected_ids, _ = _maples_partition(
            main, development_count=development_count
        )
        identifier_by_key = dict(zip(selected, selected_ids, strict=True))
        rows = _roi_rows(additional)
        member_index = {member.casefold(): member for member in main.namelist()}
        records: list[MeasurementRecord] = []
        exclusions = 0
        for key in selected:
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
                            member_index[f"train/{directory}/{key}.png".casefold()],
                        ),
                        positive_class=1,
                    )
                    for axis, directory in _AXIS_DIRECTORIES.items()
                }
                records.append(
                    MeasurementRecord(
                        instance_id=identifier_by_key[key],
                        primary=_burden(masks, fov),
                        variants={},
                        grade=None,
                        grading_split_conflict=False,
                        strict_measurement_quality=True,
                    )
                )
            except (KeyError, OSError, TypeError, ValueError):
                exclusions += 1
    return SourceMeasurements(
        name="maples_dr_messidor1",
        role="DEVELOPMENT_CALIBRATION",
        expected_input_count=development_count,
        observed_input_count=len(selected),
        records=tuple(records),
        measurement_exclusion_count=exclusions,
    )


def load_ddr_development_measurements(root: str | Path) -> SourceMeasurements:
    """Measure DDR lesion official train while leaving valid/test unopened."""

    adapter = DDRAdapter(root)
    validation = adapter.validate()
    if validation.status is not AdapterStatus.READY:
        raise ValueError("DDR adapter must be ready")
    samples = tuple(
        sample for sample in adapter.samples() if sample.official_split == "train"
    )
    records: list[MeasurementRecord] = []
    exclusions = 0
    for sample in samples:
        try:
            image = _disk_image(sample.image_path, rgb=True)
            fov_result = extract_ddr_retinal_fov(image)
            if fov_result.status is not FOVStatus.VALID or fov_result.mask is None:
                raise ValueError("DDR primary FOV is invalid")
            if fov_result.area_ratio is None or not math.isfinite(fov_result.area_ratio):
                raise ValueError("DDR FOV ratio is unavailable")
            masks = {
                axis: _binary_mask(
                    _disk_image(sample.lesion_masks[axis], rgb=False),
                    positive_class=255,
                )
                for axis in ("ma", "hem", "ex", "cws")
            }
            records.append(
                MeasurementRecord(
                    instance_id=sample.sample_id,
                    primary=_burden(masks, fov_result.mask),
                    variants={},
                    grade=None,
                    grading_split_conflict=False,
                    strict_measurement_quality=True,
                )
            )
        except (KeyError, OSError, TypeError, ValueError):
            exclusions += 1
    return SourceMeasurements(
        name="ddr",
        role="DEVELOPMENT_CALIBRATION",
        expected_input_count=len(samples),
        observed_input_count=len(samples),
        records=tuple(records),
        measurement_exclusion_count=exclusions,
    )


def load_maples_confirmation_measurements(
    main_archive_path: str | Path,
    additional_archive_path: str | Path,
    *,
    development_count: int,
) -> SourceMeasurements:
    """Measure only the precommitted MAPLES train confirmation partition."""

    with zipfile.ZipFile(main_archive_path) as main, zipfile.ZipFile(
        additional_archive_path
    ) as additional:
        _, selected, _, selected_ids = _maples_partition(
            main, development_count=development_count
        )
        identifier_by_key = dict(zip(selected, selected_ids, strict=True))
        rows = _roi_rows(additional)
        member_index = {member.casefold(): member for member in main.namelist()}
        records: list[MeasurementRecord] = []
        exclusions = 0
        for key in selected:
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
                            member_index[f"train/{directory}/{key}.png".casefold()],
                        ),
                        positive_class=1,
                    )
                    for axis, directory in _AXIS_DIRECTORIES.items()
                }
                records.append(
                    MeasurementRecord(
                        instance_id=identifier_by_key[key],
                        primary=_burden(masks, fov),
                        variants={},
                        grade=None,
                        grading_split_conflict=False,
                        strict_measurement_quality=True,
                    )
                )
            except (KeyError, OSError, TypeError, ValueError):
                exclusions += 1
    return SourceMeasurements(
        name="maples_dr_messidor1",
        role="FORMAL_R1_CONFIRMATION",
        expected_input_count=len(selected),
        observed_input_count=len(selected),
        records=tuple(records),
        measurement_exclusion_count=exclusions,
    )


def load_ddr_confirmation_measurements(root: str | Path) -> SourceMeasurements:
    """Measure DDR lesion official valid while leaving official test unopened."""

    adapter = DDRAdapter(root)
    validation = adapter.validate()
    if validation.status is not AdapterStatus.READY:
        raise ValueError("DDR adapter must be ready")
    samples = tuple(
        sample for sample in adapter.samples() if sample.official_split == "valid"
    )
    records: list[MeasurementRecord] = []
    exclusions = 0
    for sample in samples:
        try:
            image = _disk_image(sample.image_path, rgb=True)
            fov_result = extract_ddr_retinal_fov(image)
            if fov_result.status is not FOVStatus.VALID or fov_result.mask is None:
                raise ValueError("DDR primary FOV is invalid")
            if fov_result.area_ratio is None or not math.isfinite(fov_result.area_ratio):
                raise ValueError("DDR FOV ratio is unavailable")
            masks = {
                axis: _binary_mask(
                    _disk_image(sample.lesion_masks[axis], rgb=False),
                    positive_class=255,
                )
                for axis in ("ma", "hem", "ex", "cws")
            }
            records.append(
                MeasurementRecord(
                    instance_id=sample.sample_id,
                    primary=_burden(masks, fov_result.mask),
                    variants={},
                    grade=None,
                    grading_split_conflict=False,
                    strict_measurement_quality=True,
                )
            )
        except (KeyError, OSError, TypeError, ValueError):
            exclusions += 1
    return SourceMeasurements(
        name="ddr",
        role="FORMAL_R1_CONFIRMATION",
        expected_input_count=len(samples),
        observed_input_count=len(samples),
        records=tuple(records),
        measurement_exclusion_count=exclusions,
    )


def validate_ddr_confirmation_manifest(
    root: str | Path, private_manifest: str | Path
) -> dict[str, object]:
    """Validate only DDR lesion-valid bytes against the private full manifest."""

    expected: dict[str, tuple[int, str]] = {}
    reasons: list[str] = []
    try:
        with Path(private_manifest).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"relative_path", "size_bytes", "sha256", "status"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("private manifest schema is invalid")
            for row in reader:
                relative = row["relative_path"].replace("\\", "/")
                if not relative.casefold().startswith("lesion_segmentation/valid/"):
                    continue
                if relative in expected or row["status"] != "OK":
                    raise ValueError("private confirmation manifest row is invalid")
                expected[relative] = (int(row["size_bytes"]), row["sha256"])
        valid_root = Path(root) / "lesion_segmentation" / "valid"
        observed: dict[str, tuple[int, str]] = {}
        for path in sorted(valid_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(Path(root)).as_posix()
            observed[relative] = (path.stat().st_size, sha256_file(path))
        if not expected:
            reasons.append("DDR_CONFIRMATION_MANIFEST_SCOPE_EMPTY")
        if set(observed) != set(expected):
            reasons.append("DDR_CONFIRMATION_FILE_SET_MISMATCH")
        elif any(observed[key] != expected[key] for key in expected):
            reasons.append("DDR_CONFIRMATION_FILE_IDENTITY_MISMATCH")
    except (OSError, TypeError, ValueError):
        reasons.append("DDR_CONFIRMATION_MANIFEST_INVALID")
        observed = {}
    return {
        "valid": not reasons,
        "expected_file_count": len(expected),
        "observed_file_count": len(observed),
        "reason_codes": sorted(set(reasons)),
        "local_paths_recorded": False,
        "source_identifiers_recorded": False,
    }


__all__ = [
    "DevelopmentBoundary",
    "inspect_development_boundary",
    "load_ddr_confirmation_measurements",
    "load_ddr_development_measurements",
    "load_maples_confirmation_measurements",
    "load_maples_development_measurements",
    "validate_ddr_confirmation_manifest",
]

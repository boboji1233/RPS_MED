"""MAPLES source measurement with privacy-safe v0.8 substage diagnostics.

The scientific measurement operations are inherited from ``source_io``.  This
module only separates their orchestration into stable technical substages.  It
never records paths, archive members, source identifiers, exception messages,
row-level burdens, or relation outputs.
"""

from __future__ import annotations

import csv
import zipfile
from collections.abc import Callable, Mapping
from contextlib import ExitStack, suppress
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TypeVar

import numpy as np

from nts_med.concepts.burden import DR_LESION_AXES
from nts_med.concepts.fov import official_elliptical_roi_fov
from nts_med.data.adapters import opaque_sample_id
from nts_med.gate0.execution import (
    MeasurementRecord,
    ReaderMeasurements,
    SourceMeasurements,
)
from nts_med.gate0.source_io import (
    _AXIS_DIRECTORIES,
    _binary_mask,
    _burden,
    _maples_grades,
    _maples_keys,
    _roi_rows,
    _variant_burdens,
    _zip_image,
)
from nts_med.relations.algebra import Interval

T = TypeVar("T")


class MaplesSubstage(StrEnum):
    """Fixed, privacy-safe MAPLES measurement substages for v0.8."""

    ZIP_STRUCTURE = "MAPLES_ZIP_STRUCTURE"
    RESERVED_COUNT = "MAPLES_RESERVED_COUNT"
    ROI = "MAPLES_ROI"
    CSV_STRUCTURE = "MAPLES_CSV_STRUCTURE"
    PRIMARY_MASK_MEASUREMENT = "MAPLES_PRIMARY_MASK_MEASUREMENT"
    READER_SCOPE = "MAPLES_READER_SCOPE"
    READER_MASK_MEASUREMENT = "MAPLES_READER_MASK_MEASUREMENT"
    SOURCE_SUMMARY = "MAPLES_SOURCE_SUMMARY"


@dataclass(slots=True)
class MaplesDiagnosticState:
    """Aggregate-only state that deliberately excludes paths and identifiers."""

    current: MaplesSubstage | None = None
    completed: list[MaplesSubstage] = field(default_factory=list)
    observed_input_count: int | None = None
    roi_row_count: int | None = None
    measurement_eligible_count: int | None = None
    measurement_exclusion_count: int | None = None
    reader_scope_count: int | None = None
    reader_primary_ineligible_count: int | None = None
    reader_measurement_count: int | None = None

    def enter(self, substage: MaplesSubstage) -> None:
        self.current = substage

    def complete(self, substage: MaplesSubstage) -> None:
        if self.current is not substage:
            raise RuntimeError("MAPLES substage completion is out of order")
        self.completed.append(substage)

    def safe_diagnostics(self) -> dict[str, object]:
        return {
            "completed_substages": [stage.value for stage in self.completed],
            "last_completed_substage": (
                self.completed[-1].value if self.completed else None
            ),
            "observed_input_count": self.observed_input_count,
            "roi_row_count": self.roi_row_count,
            "measurement_eligible_count": self.measurement_eligible_count,
            "measurement_exclusion_count": self.measurement_exclusion_count,
            "reader_scope_count": self.reader_scope_count,
            "reader_primary_ineligible_count": self.reader_primary_ineligible_count,
            "reader_measurement_count": self.reader_measurement_count,
            "exception_message_recorded": False,
            "traceback_recorded": False,
            "absolute_paths_recorded": False,
            "archive_members_recorded": False,
            "source_identifiers_recorded": False,
            "row_level_measurements_recorded": False,
        }


class MaplesSourceMeasurementError(RuntimeError):
    """Sanitized MAPLES technical abort carrying only fixed codes and counts."""

    def __init__(
        self,
        *,
        substage: MaplesSubstage,
        error_family: str,
        diagnostics: Mapping[str, object],
    ) -> None:
        super().__init__("MAPLES_SOURCE_MEASUREMENT_SUBSTAGE_ABORT")
        self.substage = substage
        self.error_family = error_family
        self.diagnostics = dict(diagnostics)


@dataclass(frozen=True, slots=True)
class MaplesMeasurementOutcome:
    """In-memory measurements plus aggregate-only technical diagnostics."""

    measurements: SourceMeasurements
    diagnostics: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _MaplesStructure:
    main: zipfile.ZipFile
    additional: zipfile.ZipFile
    variability: zipfile.ZipFile
    keys: tuple[str, ...]
    member_index: Mapping[str, str]
    variability_index: Mapping[str, str]


def _maples_error_family(error: Exception) -> str:
    if isinstance(error, zipfile.BadZipFile):
        return "ARCHIVE_ERROR"
    if isinstance(error, csv.Error):
        return "CSV_ERROR"
    if isinstance(error, PermissionError):
        return "PERMISSION_ERROR"
    if isinstance(error, KeyError):
        return "MISSING_MEMBER_ERROR"
    if isinstance(error, TypeError):
        return "TYPE_ERROR"
    if isinstance(error, ValueError):
        return "VALUE_ERROR"
    if isinstance(error, OSError):
        return "IO_ERROR"
    return "UNEXPECTED_ERROR"


def _run_substage(
    state: MaplesDiagnosticState,
    substage: MaplesSubstage,
    operation: Callable[[], T],
) -> T:
    """Run one operation and discard all exception detail on failure."""

    state.enter(substage)
    try:
        value = operation()
    except Exception as error:
        raise MaplesSourceMeasurementError(
            substage=substage,
            error_family=_maples_error_family(error),
            diagnostics=state.safe_diagnostics(),
        ) from None
    state.complete(substage)
    return value


def _open_and_read_structure(
    stack: ExitStack,
    main_archive_path: str | Path,
    additional_archive_path: str | Path,
    variability_archive_path: str | Path,
) -> _MaplesStructure:
    main = stack.enter_context(zipfile.ZipFile(main_archive_path))
    additional = stack.enter_context(zipfile.ZipFile(additional_archive_path))
    variability = stack.enter_context(zipfile.ZipFile(variability_archive_path))
    keys = _maples_keys(main, "test")
    return _MaplesStructure(
        main=main,
        additional=additional,
        variability=variability,
        keys=keys,
        member_index={member.casefold(): member for member in main.namelist()},
        variability_index={
            member.casefold(): member for member in variability.namelist()
        },
    )


def _validate_reserved_count(
    structure: _MaplesStructure,
    state: MaplesDiagnosticState,
) -> tuple[str, ...]:
    state.observed_input_count = len(structure.keys)
    if len(structure.keys) != 60:
        raise ValueError("MAPLES reserved count is invalid")
    return structure.keys


def _read_roi_rows(
    structure: _MaplesStructure,
    state: MaplesDiagnosticState,
) -> dict[str, dict[str, int]]:
    rows = _roi_rows(structure.additional)
    state.roi_row_count = len(rows)
    return rows


def _read_diagnosis_csv(structure: _MaplesStructure) -> dict[str, int]:
    return _maples_grades(structure.main, "test")


def _measure_primary_masks(
    structure: _MaplesStructure,
    rows: Mapping[str, Mapping[str, int]],
    grades: Mapping[str, int],
    *,
    fov_bounds: tuple[float, float],
    state: MaplesDiagnosticState,
) -> tuple[list[MeasurementRecord], dict[str, str], int]:
    records: list[MeasurementRecord] = []
    key_to_id: dict[str, str] = {}
    exclusions = 0
    for key in structure.keys:
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
                        structure.main,
                        structure.member_index[
                            f"test/{directory}/{key}.png".casefold()
                        ],
                    ),
                    positive_class=1,
                )
                for axis, directory in _AXIS_DIRECTORIES.items()
            }
            ratio = float(np.mean(fov))
            primary, variants = _variant_burdens(
                masks, fov, fov_ratio=ratio, fov_bounds=fov_bounds
            )
            identifier = opaque_sample_id("maples_dr", f"annotations/test/{key}")
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
    state.measurement_eligible_count = len(records)
    state.measurement_exclusion_count = exclusions
    return records, key_to_id, exclusions


def _validate_reader_scope(
    structure: _MaplesStructure,
    key_to_id: Mapping[str, str],
    state: MaplesDiagnosticState,
) -> tuple[str, ...]:
    reader_keys = tuple(
        sorted(
            key
            for key in structure.keys
            if all(
                f"manual/{reader}/{directory}/{key}.png".casefold()
                in structure.variability_index
                for reader in ("S1", "S2", "S3")
                for directory in _AXIS_DIRECTORIES.values()
            )
        )
    )
    state.reader_scope_count = len(reader_keys)
    if len(reader_keys) != 16:
        raise ValueError("MAPLES reader scope count is invalid")
    ineligible_count = sum(key not in key_to_id for key in reader_keys)
    state.reader_primary_ineligible_count = ineligible_count
    if ineligible_count:
        raise ValueError("MAPLES reader scope is not primary eligible")
    return reader_keys


def _measure_reader_masks(
    structure: _MaplesStructure,
    reader_keys: tuple[str, ...],
    key_to_id: Mapping[str, str],
    state: MaplesDiagnosticState,
) -> ReaderMeasurements:
    reader_fov = official_elliptical_roi_fov(
        (1500, 1500), x0=0, y0=0, x1=1500, y1=1500
    )
    point_variants: dict[str, dict[str, Mapping[str, float]]] = {
        reader: {} for reader in ("S1", "S2", "S3")
    }
    envelopes: dict[str, dict[str, Interval]] = {}
    state.reader_measurement_count = 0
    for key in reader_keys:
        reader_identifier = key_to_id[key]
        by_reader: dict[str, dict[str, float]] = {}
        for reader in ("S1", "S2", "S3"):
            masks = {
                axis: _binary_mask(
                    _zip_image(
                        structure.variability,
                        structure.variability_index[
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
        state.reader_measurement_count += 1
    return ReaderMeasurements(point_variants=point_variants, envelopes=envelopes)


def _form_source_summary(
    structure: _MaplesStructure,
    records: list[MeasurementRecord],
    exclusions: int,
    reader: ReaderMeasurements,
) -> SourceMeasurements:
    return SourceMeasurements(
        name="maples_dr_messidor1",
        role="CONFIRMATORY",
        expected_input_count=60,
        observed_input_count=len(structure.keys),
        records=tuple(records),
        measurement_exclusion_count=exclusions,
        reader=reader,
    )


def load_maples_test_measurements_v08(
    main_archive_path: str | Path,
    additional_archive_path: str | Path,
    variability_archive_path: str | Path,
    *,
    fov_bounds: tuple[float, float],
) -> MaplesMeasurementOutcome:
    """Measure the frozen MAPLES scope after a future v0.8 authorization."""

    state = MaplesDiagnosticState()
    stack = ExitStack()
    try:
        structure = _run_substage(
            state,
            MaplesSubstage.ZIP_STRUCTURE,
            lambda: _open_and_read_structure(
                stack,
                main_archive_path,
                additional_archive_path,
                variability_archive_path,
            ),
        )
        _run_substage(
            state,
            MaplesSubstage.RESERVED_COUNT,
            lambda: _validate_reserved_count(structure, state),
        )
        rows = _run_substage(
            state,
            MaplesSubstage.ROI,
            lambda: _read_roi_rows(structure, state),
        )
        grades = _run_substage(
            state,
            MaplesSubstage.CSV_STRUCTURE,
            lambda: _read_diagnosis_csv(structure),
        )
        records, key_to_id, exclusions = _run_substage(
            state,
            MaplesSubstage.PRIMARY_MASK_MEASUREMENT,
            lambda: _measure_primary_masks(
                structure,
                rows,
                grades,
                fov_bounds=fov_bounds,
                state=state,
            ),
        )
        reader_keys = _run_substage(
            state,
            MaplesSubstage.READER_SCOPE,
            lambda: _validate_reader_scope(structure, key_to_id, state),
        )
        reader = _run_substage(
            state,
            MaplesSubstage.READER_MASK_MEASUREMENT,
            lambda: _measure_reader_masks(structure, reader_keys, key_to_id, state),
        )
        measurements = _run_substage(
            state,
            MaplesSubstage.SOURCE_SUMMARY,
            lambda: _form_source_summary(structure, records, exclusions, reader),
        )
        return MaplesMeasurementOutcome(
            measurements=measurements,
            diagnostics=state.safe_diagnostics(),
        )
    finally:
        with suppress(Exception):
            stack.close()


__all__ = [
    "MaplesDiagnosticState",
    "MaplesMeasurementOutcome",
    "MaplesSourceMeasurementError",
    "MaplesSubstage",
    "load_maples_test_measurements_v08",
]

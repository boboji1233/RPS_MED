"""MAPLES v0.9 source measurement with an evidence-bound grade decoder.

Version 0.9 changes only the technical decoding of MAPLES ``diagnosis.csv``.
The official release encodes diabetic-retinopathy grades as ``R0`` through
``R4A`` rather than as bare integers.  All measurement, reader, burden, and
aggregation operations remain the frozen v0.8/v0.6 operations.
"""

from __future__ import annotations

import csv
import io
import zipfile
from contextlib import ExitStack, suppress
from pathlib import Path, PurePosixPath

from nts_med.gate0.source_io_v08 import (
    MaplesDiagnosticState,
    MaplesMeasurementOutcome,
    MaplesSubstage,
    _form_source_summary,
    _measure_primary_masks,
    _measure_reader_masks,
    _open_and_read_structure,
    _read_roi_rows,
    _run_substage,
    _validate_reader_scope,
    _validate_reserved_count,
)

# Source: official MAPLES-DR dataset documentation and Scientific Data paper.
# R4S (stable treated proliferative) and R6 (ungradable) are deliberately not
# coerced into the frozen 0--4 severity contract.
MAPLES_DR_ACTIVE_GRADE_TO_SEVERITY: dict[str, int] = {
    "R0": 0,
    "R1": 1,
    "R2": 2,
    "R3": 3,
    "R4A": 4,
}


def decode_maples_dr_grade_v09(value: str) -> int:
    """Decode one official active-disease grade, failing closed otherwise."""

    token = value.strip()
    try:
        return MAPLES_DR_ACTIVE_GRADE_TO_SEVERITY[token]
    except KeyError:
        raise ValueError("MAPLES DR grade token is outside the frozen contract") from None


def maples_grades_v09(
    archive: zipfile.ZipFile,
    split: str,
) -> dict[str, int]:
    """Read an official MAPLES diagnosis table without exposing row identifiers."""

    with (
        archive.open(f"{split}/diagnosis.csv") as binary,
        io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as stream,
    ):
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("MAPLES diagnosis table is missing a header")
        normalized = {field.strip().casefold(): field for field in reader.fieldnames}
        name_field = normalized.get("name")
        grade_field = normalized.get("dr")
        if name_field is None or grade_field is None:
            raise ValueError("MAPLES diagnosis table schema is invalid")
        grades: dict[str, int] = {}
        for row in reader:
            raw_name = row.get(name_field)
            raw_grade = row.get(grade_field)
            if raw_name is None or raw_grade is None:
                raise ValueError("MAPLES diagnosis row is incomplete")
            key = PurePosixPath(raw_name.strip()).stem.casefold()
            grade = decode_maples_dr_grade_v09(raw_grade)
            if not key or key in grades:
                raise ValueError("MAPLES diagnosis row key is invalid")
            grades[key] = grade
    return grades


def load_maples_test_measurements_v09(
    main_archive_path: str | Path,
    additional_archive_path: str | Path,
    variability_archive_path: str | Path,
    *,
    fov_bounds: tuple[float, float],
) -> MaplesMeasurementOutcome:
    """Measure MAPLES using the v0.9 decoder after a future authorization."""

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
            lambda: maples_grades_v09(structure.main, "test"),
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
    "MAPLES_DR_ACTIVE_GRADE_TO_SEVERITY",
    "decode_maples_dr_grade_v09",
    "load_maples_test_measurements_v09",
    "maples_grades_v09",
]

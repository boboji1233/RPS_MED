"""Read-only MAPLES-DR v3 annotation adapter with an external image gate."""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xlrd  # type: ignore[import-untyped]
import yaml
from PIL import Image, UnidentifiedImageError

from nts_med.data.adapters.base import (
    AdapterStatus,
    DatasetSample,
    DiscoveryResult,
    ValidationResult,
    opaque_sample_id,
    require_ready,
    unavailable_discovery,
)

_SPLITS = ("train", "test")
_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
_CORE_AXIS_DIRECTORY_NAMES: Mapping[str, frozenset[str]] = {
    "ma": frozenset({"microaneurysm", "microaneurysms"}),
    "hem": frozenset({"haemorrhage", "haemorrhages", "hemorrhage", "hemorrhages"}),
    "ex": frozenset({"exudate", "exudates", "hardexudate", "hardexudates"}),
    "cws": frozenset({"cottonwoolspot", "cottonwoolspots", "softexudate", "softexudates"}),
}
_EXPECTED_BIOMARKERS = frozenset(
    {
        "opticdisc",
        "opticcup",
        "macula",
        "vessels",
        "brightuncertains",
        "exudates",
        "cottonwoolspots",
        "drusens",
        "reduncertains",
        "neovascularization",
        "microaneurysms",
        "hemorrhages",
    }
)
_REQUIRED_ADDITIONAL_FILES = (
    "biomarkers_annotation_infos.xls",
    "dataset_record.yaml",
    "diagnosis_infos.xls",
    "MESSIDOR-ROIs.csv",
)
_ROI_COLUMNS = ("name", "x0", "y0", "x1", "y1", "H", "W")
_OFFICIAL_V3_PROFILE: Mapping[str, object] = {
    "expected_train_count": 138,
    "expected_test_count": 60,
    "expected_duplicate_pairs": 2,
    "expected_roi_rows": 200,
    "roi_max_side_difference": 2,
}


@dataclass(frozen=True, slots=True)
class _MaplesReleaseProfile:
    train_count: int
    test_count: int
    duplicate_pairs: int
    roi_rows: int
    roi_max_side_difference: int


def _release_profile(value: Mapping[str, object] | None) -> _MaplesReleaseProfile:
    raw = _OFFICIAL_V3_PROFILE if value is None else value
    names = (
        "expected_train_count",
        "expected_test_count",
        "expected_duplicate_pairs",
        "expected_roi_rows",
        "roi_max_side_difference",
    )
    parsed: dict[str, int] = {}
    for name in names:
        item = raw.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError("MAPLES-DR release profile requires non-negative integer counts")
        parsed[name] = item
    if not parsed["expected_train_count"] or not parsed["expected_test_count"]:
        raise ValueError("MAPLES-DR release profile split counts must be positive")
    return _MaplesReleaseProfile(
        train_count=parsed["expected_train_count"],
        test_count=parsed["expected_test_count"],
        duplicate_pairs=parsed["expected_duplicate_pairs"],
        roi_rows=parsed["expected_roi_rows"],
        roi_max_side_difference=parsed["roi_max_side_difference"],
    )


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold()).lstrip("0123456789")


def _safe_scalar(value: str) -> str | int | float | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        try:
            return float(stripped)
        except ValueError:
            return stripped


def _source_key(value: object) -> str | None:
    """Return a comparison key without normalising away leading digits."""

    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if (
        not stripped
        or stripped in {".", ".."}
        or "/" in stripped
        or "\\" in stripped
        or Path(stripped).name != stripped
    ):
        return None
    stem = Path(stripped).stem.casefold()
    return stem if stem else None


def _parse_integer(value: object) -> int | None:
    if not isinstance(value, str) or not re.fullmatch(r"[+-]?\d+", value.strip()):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _source_key_set(value: object) -> tuple[frozenset[str], int, int]:
    if not isinstance(value, list):
        return frozenset(), 1, 0
    parsed = [_source_key(item) for item in value]
    invalid = sum(item is None for item in parsed)
    keys = [item for item in parsed if item is not None]
    duplicates = len(keys) - len(set(keys))
    return frozenset(keys), invalid, duplicates


def _normalised_label_set(value: object) -> tuple[frozenset[str], int, int]:
    if not isinstance(value, list):
        return frozenset(), 1, 0
    labels = [
        _normalise_name(item) if isinstance(item, str) and item.strip() else ""
        for item in value
    ]
    invalid = sum(not item for item in labels)
    valid = [item for item in labels if item]
    duplicates = len(valid) - len(set(valid))
    return frozenset(valid), invalid, duplicates


def _parse_roi_table(
    path: Path,
    *,
    max_side_difference: int,
) -> tuple[dict[str, tuple[int, int]], int, tuple[str, ...]]:
    dimensions: dict[str, tuple[int, int]] = {}
    row_count = 0
    malformed = 0
    duplicates = 0
    invalid_bounds = 0
    invalid_shape = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(_ROI_COLUMNS).issubset(reader.fieldnames):
            return {}, 0, ("MAPLES_DR_ROI_SCHEMA_INVALID",)
        for row in reader:
            row_count += 1
            key = _source_key(row.get("name"))
            values = tuple(_parse_integer(row.get(column)) for column in _ROI_COLUMNS[1:])
            x0, y0, x1, y1, height, width = values
            if (
                key is None
                or x0 is None
                or y0 is None
                or x1 is None
                or y1 is None
                or height is None
                or width is None
            ):
                malformed += 1
                continue
            if (
                width <= 0
                or height <= 0
                or not 0 <= x0 < x1 <= width
                or not 0 <= y0 < y1 <= height
            ):
                invalid_bounds += 1
                continue
            if abs((x1 - x0) - (y1 - y0)) > max_side_difference:
                invalid_shape += 1
                continue
            if key in dimensions:
                duplicates += 1
                continue
            dimensions[key] = (width, height)
    issues: list[str] = []
    if malformed:
        issues.append(f"MAPLES_DR_ROI_ROWS_INVALID:{malformed}")
    if invalid_bounds:
        issues.append(f"MAPLES_DR_ROI_BOUNDS_INVALID:{invalid_bounds}")
    if invalid_shape:
        issues.append(f"MAPLES_DR_ROI_SHAPE_INVALID:{invalid_shape}")
    if duplicates:
        issues.append(f"MAPLES_DR_ROI_KEYS_DUPLICATED:{duplicates}")
    if not dimensions:
        issues.append("MAPLES_DR_ROI_ROWS_EMPTY")
    return dimensions, row_count, tuple(issues)


def _workbook_is_readable(path: Path) -> bool:
    try:
        workbook = xlrd.open_workbook(str(path), on_demand=True)
    except (OSError, xlrd.XLRDError):
        return False
    readable = False
    try:
        sheets = workbook.sheets()
        readable = bool(sheets) and all(
            sheet.nrows > 0 and sheet.ncols > 0 for sheet in sheets
        )
    except (IndexError, OSError, TypeError, ValueError, xlrd.XLRDError):
        readable = False
    finally:
        try:
            workbook.release_resources()
        except (AttributeError, OSError, TypeError, ValueError, xlrd.XLRDError):
            readable = False
    return readable


def _find_file(parent: Path, normalised_name: str) -> Path | None:
    matches = [
        child
        for child in parent.iterdir()
        if child.is_file() and _normalise_name(child.name) == normalised_name
    ]
    return sorted(matches, key=lambda item: item.name.casefold())[0] if len(matches) == 1 else None


def _find_axis_directory(split_root: Path, axis: str) -> Path | None:
    matches = [
        child
        for child in split_root.iterdir()
        if child.is_dir() and _normalise_name(child.name) in _CORE_AXIS_DIRECTORY_NAMES[axis]
    ]
    return sorted(matches, key=lambda item: item.name.casefold())[0] if len(matches) == 1 else None


def _mask_map(directory: Path) -> tuple[dict[str, Path], int]:
    result: dict[str, Path] = {}
    duplicates = 0
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.casefold() not in _IMAGE_SUFFIXES:
            continue
        key = path.stem.casefold()
        if key in result:
            duplicates += 1
        else:
            result[key] = path
    return result, duplicates


def _parse_diagnosis(
    path: Path,
) -> tuple[dict[str, tuple[str | int | float | None, str | int | float | None]], int, int]:
    rows: dict[str, tuple[str | int | float | None, str | int | float | None]] = {}
    malformed = 0
    duplicates = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            return rows, 1, 0
        headers = {_normalise_name(name): name for name in reader.fieldnames}
        name_column = headers.get("name")
        dr_column = headers.get("dr")
        me_column = headers.get("me")
        if name_column is None or dr_column is None or me_column is None:
            return rows, 1, 0
        for row in reader:
            key = _source_key(row.get(name_column, ""))
            if key is None:
                malformed += 1
            elif key in rows:
                duplicates += 1
            else:
                rows[key] = (
                    _safe_scalar(row.get(dr_column, "")),
                    _safe_scalar(row.get(me_column, "")),
                )
    return rows, malformed, duplicates


def _recursive_image_map(root: Path) -> tuple[dict[str, Path], int]:
    result: dict[str, Path] = {}
    duplicates = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if not path.is_file() or path.suffix.casefold() not in _IMAGE_SUFFIXES:
            continue
        key = path.stem.casefold()
        if key in result:
            duplicates += 1
        else:
            result[key] = path
    return result, duplicates


def _independent_roots(annotation_root: Path, image_root: Path) -> bool:
    annotation = annotation_root.resolve(strict=False)
    images = image_root.resolve(strict=False)
    if annotation == images:
        return False
    try:
        annotation.relative_to(images)
        return False
    except ValueError:
        pass
    try:
        images.relative_to(annotation)
        return False
    except ValueError:
        return True


@dataclass(frozen=True, slots=True)
class _AdditionalMetadataInspection:
    status: AdapterStatus
    issues: tuple[str, ...]
    root_available: bool
    required_files_present: int
    split_keys: Mapping[str, frozenset[str]]
    split_matches_annotations: bool | None
    roi_dimensions: Mapping[str, tuple[int, int]]
    roi_row_count: int | None
    duplicate_pair_count: int | None
    expected_missing_by_category: Mapping[str, frozenset[str]]


@dataclass(frozen=True, slots=True)
class _MaplesInspection:
    status: AdapterStatus
    issues: tuple[str, ...]
    diagnoses: Mapping[
        str,
        Mapping[str, tuple[str | int | float | None, str | int | float | None]],
    ]
    masks: Mapping[str, Mapping[str, Mapping[str, Path]]]
    all_mask_category_counts: Mapping[str, Mapping[str, int]]
    all_mask_category_keys: Mapping[str, Mapping[str, frozenset[str]]]
    images: Mapping[str, Path]
    additional_metadata: _AdditionalMetadataInspection
    image_root_available: bool
    image_root_qualified: bool
    image_unreadable_count: int
    image_non_rgb_count: int
    image_size_mismatch_count: int


class MaplesDRAdapter:
    """Expose MAPLES-DR samples only with a separate MESSIDOR-1 image root."""

    dataset_name = "maples_dr"

    def __init__(
        self,
        annotation_root: str | Path,
        image_root: str | Path | None = None,
        metadata_root: str | Path | None = None,
        release_profile: Mapping[str, object] | None = None,
    ) -> None:
        self.annotation_root = Path(annotation_root)
        self.image_root = Path(image_root) if image_root is not None else None
        self.metadata_root = Path(metadata_root) if metadata_root is not None else None
        self.release_profile = _release_profile(release_profile)

    def _inspect_additional_metadata(
        self,
        annotation_split_keys: Mapping[str, set[str]],
    ) -> _AdditionalMetadataInspection:
        if self.metadata_root is None or not self.metadata_root.is_dir():
            return _AdditionalMetadataInspection(
                status=AdapterStatus.INCOMPLETE,
                issues=("MAPLES_DR_OFFICIAL_ADDITIONAL_METADATA_REQUIRED",),
                root_available=False,
                required_files_present=0,
                split_keys={},
                split_matches_annotations=None,
                roi_dimensions={},
                roi_row_count=None,
                duplicate_pair_count=None,
                expected_missing_by_category={},
            )

        files: dict[str, Path] = {}
        for expected in _REQUIRED_ADDITIONAL_FILES:
            try:
                matched = _find_file(self.metadata_root, _normalise_name(expected))
            except OSError:
                return _AdditionalMetadataInspection(
                    status=AdapterStatus.INVALID,
                    issues=("MAPLES_DR_ADDITIONAL_METADATA_READ_ERROR",),
                    root_available=True,
                    required_files_present=len(files),
                    split_keys={},
                    split_matches_annotations=None,
                    roi_dimensions={},
                    roi_row_count=None,
                    duplicate_pair_count=None,
                    expected_missing_by_category={},
                )
            if matched is not None and matched.stat().st_size > 0:
                files[expected] = matched
        if len(files) != len(_REQUIRED_ADDITIONAL_FILES):
            return _AdditionalMetadataInspection(
                status=AdapterStatus.INCOMPLETE,
                issues=(
                    "MAPLES_DR_ADDITIONAL_METADATA_FILES_MISSING:"
                    f"{len(_REQUIRED_ADDITIONAL_FILES) - len(files)}",
                ),
                root_available=True,
                required_files_present=len(files),
                split_keys={},
                split_matches_annotations=None,
                roi_dimensions={},
                roi_row_count=None,
                duplicate_pair_count=None,
                expected_missing_by_category={},
            )

        issues: list[str] = []
        invalid_workbooks = sum(
            not _workbook_is_readable(files[name])
            for name in ("biomarkers_annotation_infos.xls", "diagnosis_infos.xls")
        )
        if invalid_workbooks:
            issues.append(f"MAPLES_DR_ADDITIONAL_WORKBOOKS_INVALID:{invalid_workbooks}")
        try:
            loaded: Any = yaml.safe_load(files["dataset_record.yaml"].read_text("utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError):
            loaded = None
            issues.append("MAPLES_DR_DATASET_RECORD_READ_ERROR")
        if not isinstance(loaded, Mapping):
            issues.append("MAPLES_DR_DATASET_RECORD_SCHEMA_INVALID")
            loaded = {}

        biomarkers, invalid_biomarkers, duplicate_biomarkers = _normalised_label_set(
            loaded.get("biomarkers")
        )
        if (
            invalid_biomarkers
            or duplicate_biomarkers
            or biomarkers != _EXPECTED_BIOMARKERS
        ):
            issues.append("MAPLES_DR_DATASET_RECORD_BIOMARKERS_INVALID")

        record_splits: dict[str, frozenset[str]] = {}
        for split in _SPLITS:
            keys, invalid_keys, duplicate_keys = _source_key_set(loaded.get(split))
            record_splits[split] = keys
            if invalid_keys or duplicate_keys or not keys:
                issues.append(
                    f"MAPLES_DR_DATASET_RECORD_SPLIT_INVALID:{split.upper()}"
                )
        expected_split_counts = {
            "train": self.release_profile.train_count,
            "test": self.release_profile.test_count,
        }
        for split, expected_count in expected_split_counts.items():
            if len(record_splits[split]) != expected_count:
                issues.append(
                    "MAPLES_DR_RELEASE_PROFILE_SPLIT_COUNT_MISMATCH:"
                    f"{split.upper()}"
                )
        split_overlap = record_splits.get("train", frozenset()) & record_splits.get(
            "test", frozenset()
        )
        if split_overlap:
            issues.append(
                f"MAPLES_DR_DATASET_RECORD_SPLIT_OVERLAP:{len(split_overlap)}"
            )

        split_match: bool | None = None
        if all(split in annotation_split_keys for split in _SPLITS):
            mismatch_count = sum(
                len(record_splits[split] ^ annotation_split_keys[split])
                for split in _SPLITS
            )
            split_match = mismatch_count == 0
            if mismatch_count:
                issues.append(f"MAPLES_DR_OFFICIAL_SPLIT_MISMATCH:{mismatch_count}")

        duplicate_value = loaded.get("duplicates")
        duplicate_pairs: dict[str, str] = {}
        duplicate_mapping_valid = isinstance(duplicate_value, Mapping)
        if duplicate_mapping_valid:
            for raw_key, raw_value in duplicate_value.items():
                key = _source_key(raw_key)
                value = _source_key(raw_value)
                if key is None or value is None or key == value or key in duplicate_pairs:
                    duplicate_mapping_valid = False
                    continue
                duplicate_pairs[key] = value
        if not duplicate_pairs:
            duplicate_mapping_valid = False
        train_keys = record_splits.get("train", frozenset())
        all_split_keys = train_keys | record_splits.get("test", frozenset())
        duplicate_values = tuple(duplicate_pairs.values())
        if (
            not duplicate_mapping_valid
            or not set(duplicate_pairs).issubset(train_keys)
            or set(duplicate_values) & all_split_keys
            or len(duplicate_values) != len(set(duplicate_values))
            or set(duplicate_pairs) & set(duplicate_values)
        ):
            issues.append("MAPLES_DR_DUPLICATE_MAPPING_INVALID")
        if len(duplicate_pairs) != self.release_profile.duplicate_pairs:
            issues.append("MAPLES_DR_RELEASE_PROFILE_DUPLICATE_COUNT_MISMATCH")

        expected_missing_by_category: dict[str, frozenset[str]] = {}
        for field, category in (("no_cup", "opticcup"), ("no_macula", "macula")):
            values, invalid_values, duplicate_values_in_field = _source_key_set(
                loaded.get(field)
            )
            expected_missing_by_category[category] = values
            if (
                invalid_values
                or duplicate_values_in_field
                or not values.issubset(all_split_keys)
            ):
                issues.append(f"MAPLES_DR_DATASET_RECORD_{field.upper()}_INVALID")

        try:
            roi_dimensions, roi_count, roi_issues = _parse_roi_table(
                files["MESSIDOR-ROIs.csv"],
                max_side_difference=self.release_profile.roi_max_side_difference,
            )
        except (OSError, UnicodeError):
            roi_dimensions = {}
            roi_count = 0
            roi_issues = ("MAPLES_DR_ROI_READ_ERROR",)
        issues.extend(roi_issues)
        if roi_count != self.release_profile.roi_rows:
            issues.append("MAPLES_DR_RELEASE_PROFILE_ROI_COUNT_MISMATCH")
        expected_roi_keys = all_split_keys | set(duplicate_values)
        roi_key_mismatch = len(set(roi_dimensions) ^ expected_roi_keys)
        if roi_key_mismatch:
            issues.append(f"MAPLES_DR_ROI_KEY_MISMATCH:{roi_key_mismatch}")

        return _AdditionalMetadataInspection(
            status=AdapterStatus.INVALID if issues else AdapterStatus.READY,
            issues=tuple(sorted(set(issues))),
            root_available=True,
            required_files_present=len(files),
            split_keys=record_splits,
            split_matches_annotations=split_match,
            roi_dimensions=roi_dimensions,
            roi_row_count=roi_count,
            duplicate_pair_count=len(duplicate_pairs),
            expected_missing_by_category=expected_missing_by_category,
        )

    def discover(self) -> DiscoveryResult:
        if not self.annotation_root.exists() or not self.annotation_root.is_dir():
            return unavailable_discovery(self.annotation_root)
        return DiscoveryResult(
            status=AdapterStatus.READY,
            root=self.annotation_root,
            files=None,
            message="MAPLES_DR_ANNOTATIONS_AVAILABLE",
        )

    def _inspect(self) -> _MaplesInspection:
        if not self.annotation_root.exists() or not self.annotation_root.is_dir():
            return _MaplesInspection(
                status=AdapterStatus.DATA_NOT_AVAILABLE,
                issues=("MAPLES_DR_ANNOTATION_ROOT_UNAVAILABLE",),
                diagnoses={},
                masks={},
                all_mask_category_counts={},
                all_mask_category_keys={},
                images={},
                additional_metadata=_AdditionalMetadataInspection(
                    status=AdapterStatus.INCOMPLETE,
                    issues=("MAPLES_DR_OFFICIAL_ADDITIONAL_METADATA_REQUIRED",),
                    root_available=False,
                    required_files_present=0,
                    split_keys={},
                    split_matches_annotations=None,
                    roi_dimensions={},
                    roi_row_count=None,
                    duplicate_pair_count=None,
                    expected_missing_by_category={},
                ),
                image_root_available=False,
                image_root_qualified=False,
                image_unreadable_count=0,
                image_non_rgb_count=0,
                image_size_mismatch_count=0,
            )

        incomplete: list[str] = []
        invalid: list[str] = []
        diagnoses: dict[
            str,
            dict[str, tuple[str | int | float | None, str | int | float | None]],
        ] = {}
        masks: dict[str, dict[str, dict[str, Path]]] = {}
        all_mask_category_counts: dict[str, dict[str, int]] = {}
        all_mask_category_keys: dict[str, dict[str, frozenset[str]]] = {}
        images: dict[str, Path] = {}
        image_root_available = False
        image_root_qualified = False
        image_unreadable_count = 0
        image_non_rgb_count = 0
        image_size_mismatch_count = 0

        metadata_root_available = self.metadata_root is not None and self.metadata_root.is_dir()
        image_root_exists = self.image_root is not None and self.image_root.is_dir()
        roots_independent = True
        if metadata_root_available:
            assert self.metadata_root is not None
            if not _independent_roots(self.annotation_root, self.metadata_root):
                roots_independent = False
        if image_root_exists:
            assert self.image_root is not None
            if not _independent_roots(self.annotation_root, self.image_root):
                roots_independent = False
            if metadata_root_available:
                assert self.metadata_root is not None
                if not _independent_roots(self.metadata_root, self.image_root):
                    roots_independent = False
        if not roots_independent:
            invalid.append("MAPLES_DR_ROOTS_NOT_INDEPENDENT")

        split_keys: dict[str, set[str]] = {}
        for split in _SPLITS:
            split_root = self.annotation_root / split
            if not split_root.is_dir():
                incomplete.append(f"MAPLES_DR_SPLIT_MISSING:{split.upper()}")
                continue
            diagnoses[split] = {}
            masks[split] = {}
            all_mask_category_counts[split] = {}
            all_mask_category_keys[split] = {}
            try:
                diagnosis_file = _find_file(split_root, "diagnosiscsv")
            except OSError:
                diagnosis_file = None
                invalid.append(
                    f"MAPLES_DR_DIAGNOSIS_DISCOVERY_READ_ERROR:{split.upper()}"
                )
            if diagnosis_file is None:
                incomplete.append(f"MAPLES_DR_DIAGNOSIS_MISSING:{split.upper()}")
            else:
                try:
                    split_diagnoses, malformed, duplicate_diagnoses = _parse_diagnosis(
                        diagnosis_file
                    )
                except (OSError, UnicodeError):
                    invalid.append(f"MAPLES_DR_DIAGNOSIS_READ_ERROR:{split.upper()}")
                else:
                    diagnoses[split] = split_diagnoses
                    if malformed:
                        invalid.append(
                            "MAPLES_DR_DIAGNOSIS_ROWS_INVALID:"
                            f"{split.upper()}:{malformed}"
                        )
                    if duplicate_diagnoses:
                        invalid.append(
                            "MAPLES_DR_DIAGNOSIS_KEY_DUPLICATES:"
                            f"{split.upper()}:{duplicate_diagnoses}"
                        )
                    if not split_diagnoses:
                        incomplete.append(
                            f"MAPLES_DR_DIAGNOSIS_EMPTY:{split.upper()}"
                        )
            try:
                category_directories = sorted(
                    (child for child in split_root.iterdir() if child.is_dir()),
                    key=lambda item: item.name.casefold(),
                )
            except OSError:
                invalid.append(f"MAPLES_DR_SPLIT_READ_ERROR:{split.upper()}")
                continue
            for category_directory in category_directories:
                category = _normalise_name(category_directory.name)
                try:
                    category_files, duplicate_category_files = _mask_map(
                        category_directory
                    )
                except OSError:
                    invalid.append(
                        f"MAPLES_DR_MASK_CATEGORY_READ_ERROR:{split.upper()}:{category}"
                    )
                    continue
                all_mask_category_counts[split][category] = len(category_files)
                all_mask_category_keys[split][category] = frozenset(category_files)
                if duplicate_category_files:
                    invalid.append(
                        "MAPLES_DR_MASK_CATEGORY_KEY_DUPLICATES:"
                        f"{split.upper()}:{category}:{duplicate_category_files}"
                    )
            for axis in _CORE_AXIS_DIRECTORY_NAMES:
                try:
                    axis_directory = _find_axis_directory(split_root, axis)
                except OSError:
                    axis_directory = None
                    invalid.append(
                        f"MAPLES_DR_AXIS_DISCOVERY_READ_ERROR:{split.upper()}:{axis.upper()}"
                    )
                if axis_directory is None:
                    incomplete.append(
                        f"MAPLES_DR_CORE_AXIS_MISSING:{split.upper()}:{axis.upper()}"
                    )
                    continue
                try:
                    axis_masks, duplicate_masks = _mask_map(axis_directory)
                except OSError:
                    invalid.append(
                        f"MAPLES_DR_AXIS_READ_ERROR:{split.upper()}:{axis.upper()}"
                    )
                    continue
                masks[split][axis] = axis_masks
                if duplicate_masks:
                    invalid.append(
                        "MAPLES_DR_CORE_MASK_KEY_DUPLICATES:"
                        f"{split.upper()}:{axis.upper()}:{duplicate_masks}"
                    )

            available_axes = [
                set(masks[split][axis])
                for axis in _CORE_AXIS_DIRECTORY_NAMES
                if axis in masks[split]
            ]
            if len(available_axes) == len(_CORE_AXIS_DIRECTORY_NAMES):
                union = set().union(*available_axes)
                intersection = set.intersection(*available_axes)
                if union != intersection:
                    for axis in _CORE_AXIS_DIRECTORY_NAMES:
                        missing = len(union - set(masks[split][axis]))
                        if missing:
                            invalid.append(
                                "MAPLES_DR_CORE_MASK_MISSING:"
                                f"{split.upper()}:{axis.upper()}:{missing}"
                            )
                split_keys[split] = intersection

        overlap = len(split_keys.get("train", set()) & split_keys.get("test", set()))
        if overlap:
            invalid.append(f"MAPLES_DR_CROSS_SPLIT_KEY_OVERLAP:{overlap}")
        for split, annotation_keys in split_keys.items():
            diagnosis_keys = set(diagnoses.get(split, {}))
            missing_diagnoses = len(annotation_keys - diagnosis_keys)
            unused_diagnoses = len(diagnosis_keys - annotation_keys)
            if missing_diagnoses or unused_diagnoses:
                invalid.append(
                    "MAPLES_DR_MASK_DIAGNOSIS_MISMATCH:"
                    f"{split.upper()}:missing={missing_diagnoses}:"
                    f"unused={unused_diagnoses}"
                )

        annotation_instances = {
            (split, source_key)
            for split, source_keys in split_keys.items()
            for source_key in source_keys
        }

        additional_metadata = self._inspect_additional_metadata(split_keys)
        if additional_metadata.status is AdapterStatus.INVALID:
            invalid.extend(additional_metadata.issues)
        elif additional_metadata.status is AdapterStatus.INCOMPLETE:
            incomplete.extend(additional_metadata.issues)
        if all(split in additional_metadata.split_keys for split in _SPLITS):
            category_mismatch_count = 0
            for split in _SPLITS:
                official_keys = set(additional_metadata.split_keys[split])
                for category in _EXPECTED_BIOMARKERS:
                    expected_keys = official_keys - set(
                        additional_metadata.expected_missing_by_category.get(
                            category, frozenset()
                        )
                    )
                    actual_keys = set(
                        all_mask_category_keys.get(split, {}).get(
                            category, frozenset()
                        )
                    )
                    category_mismatch_count += len(expected_keys ^ actual_keys)
            if category_mismatch_count:
                invalid.append(
                    "MAPLES_DR_ALL_CATEGORY_MASK_MISMATCH:"
                    f"{category_mismatch_count}"
                )

        annotation_keys = {source_key for _, source_key in annotation_instances}
        if not image_root_exists:
            incomplete.append("MAPLES_DR_QUALIFIED_MESSIDOR1_IMAGES_REQUIRED")
        elif roots_independent:
            assert self.image_root is not None
            image_root_available = True
            duplicate_images = 0
            missing_images = len(annotation_keys)
            try:
                images, duplicate_images = _recursive_image_map(self.image_root)
            except OSError:
                invalid.append("MAPLES_DR_IMAGE_ROOT_READ_ERROR")
            else:
                if duplicate_images:
                    invalid.append(f"MAPLES_DR_IMAGE_KEY_DUPLICATES:{duplicate_images}")
                missing_images = sum(source_key not in images for source_key in annotation_keys)
                if missing_images:
                    incomplete.append(f"MAPLES_DR_IMAGE_MATCH_MISSING:{missing_images}")
                for source_key in sorted(annotation_keys & set(images)):
                    try:
                        with Image.open(images[source_key]) as image:
                            image.load()
                            image_size = (int(image.width), int(image.height))
                            image_bands = len(image.getbands())
                    except (OSError, UnidentifiedImageError, ValueError, SyntaxError):
                        image_unreadable_count += 1
                        continue
                    if image_bands != 3:
                        image_non_rgb_count += 1
                    expected_size = additional_metadata.roi_dimensions.get(source_key)
                    if expected_size is not None and image_size != expected_size:
                        image_size_mismatch_count += 1
                if image_unreadable_count:
                    invalid.append(
                        f"MAPLES_DR_IMAGE_UNREADABLE:{image_unreadable_count}"
                    )
                if image_non_rgb_count:
                    invalid.append(
                        f"MAPLES_DR_IMAGE_CHANNELS_INVALID:{image_non_rgb_count}"
                    )
                if image_size_mismatch_count:
                    invalid.append(
                        "MAPLES_DR_IMAGE_ROI_SIZE_MISMATCH:"
                        f"{image_size_mismatch_count}"
                    )
                image_root_qualified = (
                    additional_metadata.status is AdapterStatus.READY
                    and duplicate_images == 0
                    and missing_images == 0
                    and image_unreadable_count == 0
                    and image_non_rgb_count == 0
                    and image_size_mismatch_count == 0
                )

        if invalid:
            status = AdapterStatus.INVALID
        elif incomplete:
            status = AdapterStatus.INCOMPLETE
        else:
            status = AdapterStatus.READY
        return _MaplesInspection(
            status=status,
            issues=tuple(sorted(set((*invalid, *incomplete)))),
            diagnoses=diagnoses,
            masks=masks,
            all_mask_category_counts=all_mask_category_counts,
            all_mask_category_keys=all_mask_category_keys,
            images=images,
            additional_metadata=additional_metadata,
            image_root_available=image_root_available,
            image_root_qualified=image_root_qualified,
            image_unreadable_count=image_unreadable_count,
            image_non_rgb_count=image_non_rgb_count,
            image_size_mismatch_count=image_size_mismatch_count,
        )

    def validate(self) -> ValidationResult:
        inspection = self._safe_inspect()
        message = {
            AdapterStatus.READY: "MAPLES_DR_READY",
            AdapterStatus.INCOMPLETE: "MAPLES_DR_REQUIRED_COMPONENTS_INCOMPLETE",
            AdapterStatus.INVALID: "MAPLES_DR_INVALID",
            AdapterStatus.DATA_NOT_AVAILABLE: "DATA_NOT_AVAILABLE",
        }[inspection.status]
        return ValidationResult(inspection.status, inspection.issues, message)

    def samples(self) -> tuple[DatasetSample, ...]:
        inspection = self._safe_inspect()
        require_ready(
            self.dataset_name,
            ValidationResult(inspection.status, inspection.issues, "MAPLES_DR_SAMPLE_CHECK"),
        )
        samples: list[DatasetSample] = []
        for split in _SPLITS:
            split_masks = inspection.masks.get(split, {})
            keys = set.intersection(
                *(set(split_masks[axis]) for axis in _CORE_AXIS_DIRECTORY_NAMES)
            )
            for source_key in sorted(keys):
                dr_grade, me_grade = inspection.diagnoses[split][source_key]
                samples.append(
                    DatasetSample(
                        sample_id=opaque_sample_id(
                            self.dataset_name, f"annotations/{split}/{source_key}"
                        ),
                        image_path=inspection.images[source_key],
                        dataset_name=self.dataset_name,
                        official_split=split,
                        grade_fields={"dr_grade": dr_grade, "me_grade": me_grade},
                        lesion_masks={
                            axis: split_masks[axis][source_key]
                            for axis in _CORE_AXIS_DIRECTORY_NAMES
                        },
                        metadata={
                            "annotation_type": "pixel_mask",
                            "four_axis_complete": True,
                            "image_release": "MESSIDOR_1_REQUIRED",
                        },
                    )
                )
        return tuple(samples)

    def audit_facts(self) -> Mapping[str, object]:
        inspection = self._safe_inspect()
        diagnosis_values = tuple(
            values
            for split in _SPLITS
            for values in inspection.diagnoses.get(split, {}).values()
        )
        dr_distribution = Counter(values[0] for values in diagnosis_values)
        me_distribution = Counter(values[1] for values in diagnosis_values)
        diagnosis_splits: dict[str, object] = {}
        for split in _SPLITS:
            split_values = tuple(inspection.diagnoses.get(split, {}).values())
            split_dr_distribution = Counter(values[0] for values in split_values)
            split_me_distribution = Counter(values[1] for values in split_values)
            diagnosis_splits[split] = {
                "count": len(split_values),
                "dr_grade_distribution": {
                    str(value): split_dr_distribution[value]
                    for value in sorted(split_dr_distribution, key=str)
                },
                "me_grade_distribution": {
                    str(value): split_me_distribution[value]
                    for value in sorted(split_me_distribution, key=str)
                },
            }
        annotation_splits: dict[str, object] = {}
        annotation_instances: set[tuple[str, str]] = set()
        for split in _SPLITS:
            split_masks = inspection.masks.get(split, {})
            axis_keys = {
                axis: set(split_masks.get(axis, {})) for axis in _CORE_AXIS_DIRECTORY_NAMES
            }
            union = set().union(*axis_keys.values()) if axis_keys else set()
            complete = set.intersection(*axis_keys.values()) if axis_keys else set()
            annotation_instances.update((split, source_key) for source_key in complete)
            annotation_splits[split] = {
                "complete_four_axis_count": len(complete),
                "core_mask_counts": {
                    axis: len(axis_keys[axis]) for axis in _CORE_AXIS_DIRECTORY_NAMES
                },
                "core_missing_mask_counts": {
                    axis: len(union - axis_keys[axis]) for axis in _CORE_AXIS_DIRECTORY_NAMES
                },
                "all_mask_category_counts": dict(
                    sorted(inspection.all_mask_category_counts.get(split, {}).items())
                ),
                "all_mask_file_count": sum(
                    inspection.all_mask_category_counts.get(split, {}).values()
                ),
            }
        images_available = inspection.image_root_available
        metadata = inspection.additional_metadata
        return {
            "schema_version": 1,
            "dataset": self.dataset_name,
            "status": inspection.status.value,
            "issues": list(inspection.issues),
            "labels": {
                "count": len(diagnosis_values),
                "splits": diagnosis_splits,
                "dr_grade_distribution": {
                    str(value): dr_distribution[value]
                    for value in sorted(dr_distribution, key=str)
                },
                "me_grade_distribution": {
                    str(value): me_distribution[value]
                    for value in sorted(me_distribution, key=str)
                },
            },
            "annotations": {
                "splits": annotation_splits,
                "total_complete_four_axis_count": len(annotation_instances),
                "total_mask_file_count": sum(
                    sum(split_counts.values())
                    for split_counts in inspection.all_mask_category_counts.values()
                ),
            },
            "additional_metadata": {
                "status": metadata.status.value,
                "root_available": metadata.root_available,
                "required_file_count": len(_REQUIRED_ADDITIONAL_FILES),
                "required_files_present": metadata.required_files_present,
                "official_split_counts": {
                    split: len(metadata.split_keys.get(split, frozenset()))
                    for split in _SPLITS
                },
                "official_split_matches_annotations": metadata.split_matches_annotations,
                "roi_row_count": metadata.roi_row_count,
                "known_duplicate_pair_count": metadata.duplicate_pair_count,
            },
            "images": {
                "root_available": images_available,
                "qualified_root_available": inspection.image_root_qualified,
                "matched_count": (
                    sum(
                        source_key in inspection.images
                        for _, source_key in annotation_instances
                    )
                    if images_available
                    else None
                ),
                "missing_count": sum(
                    source_key not in inspection.images
                    for _, source_key in annotation_instances
                )
                if images_available
                else len(annotation_instances),
                "unreadable_count": inspection.image_unreadable_count,
                "non_rgb_count": inspection.image_non_rgb_count,
                "roi_size_mismatch_count": inspection.image_size_mismatch_count,
            },
        }

    def _safe_inspect(self) -> _MaplesInspection:
        try:
            return self._inspect()
        except (OSError, UnicodeError):
            return _MaplesInspection(
                status=AdapterStatus.INVALID,
                issues=("MAPLES_DR_ADAPTER_READ_ERROR",),
                diagnoses={},
                masks={},
                all_mask_category_counts={},
                all_mask_category_keys={},
                images={},
                additional_metadata=_AdditionalMetadataInspection(
                    status=AdapterStatus.INVALID,
                    issues=("MAPLES_DR_ADDITIONAL_METADATA_READ_ERROR",),
                    root_available=False,
                    required_files_present=0,
                    split_keys={},
                    split_matches_annotations=None,
                    roi_dimensions={},
                    roi_row_count=None,
                    duplicate_pair_count=None,
                    expected_missing_by_category={},
                ),
                image_root_available=False,
                image_root_qualified=False,
                image_unreadable_count=0,
                image_non_rgb_count=0,
                image_size_mismatch_count=0,
            )


__all__ = ["MaplesDRAdapter"]

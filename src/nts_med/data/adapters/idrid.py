"""Read-only adapter for the official IDRiD segmentation and grading tasks."""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nts_med.data.adapters.base import (
    AdapterStatus,
    DatasetSample,
    DiscoveryResult,
    ValidationResult,
    opaque_sample_id,
    require_ready,
    unavailable_discovery,
)
from nts_med.data.manifests import sha256_file

_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
_AXIS_NAMES: Mapping[str, frozenset[str]] = {
    "ma": frozenset({"microaneurysm", "microaneurysms"}),
    "hem": frozenset({"haemorrhage", "haemorrhages", "hemorrhage", "hemorrhages"}),
    "ex": frozenset({"hardexudate", "hardexudates"}),
    "cws": frozenset({"softexudate", "softexudates", "cottonwoolspot", "cottonwoolspots"}),
}
_AXIS_SUFFIXES: Mapping[str, tuple[str, ...]] = {
    "ma": ("_ma",),
    "hem": ("_he", "_hem"),
    "ex": ("_ex",),
    "cws": ("_se", "_cws"),
}
_SPLIT_NAMES: Mapping[str, frozenset[str]] = {
    "train": frozenset({"train", "trainingset", "atrainingset"}),
    "test": frozenset({"test", "testingset", "btestingset"}),
}
_GRADING_COMPONENT = "B. Disease Grading"
_GRADING_ARCHIVE_NAME = f"{_GRADING_COMPONENT}.zip"
_OFFICIAL_RECORD_IDENTIFIERS = frozenset(
    {
        "10.21227/h25w98",
        "http://doi.org/10.21227/h25w98",
        "https://doi.org/10.21227/h25w98",
    }
)
_OFFICIAL_DISTRIBUTION = "IEEE_DATAPORT_OPEN_ACCESS_REGISTERED"
_ALLOWED_ACQUISITION_METHODS = frozenset(
    {
        "USER_MANUAL_DOWNLOAD",
        "USER_MANUAL_DOWNLOAD_AFTER_REGISTERED_OFFICIAL_WORKFLOW",
    }
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold()).lstrip("0123456789")


def _find_child(parent: Path, accepted: frozenset[str]) -> Path | None:
    if not parent.is_dir():
        return None
    matches = [
        child
        for child in parent.iterdir()
        if child.is_dir() and _normalise_name(child.name) in accepted
    ]
    return sorted(matches, key=lambda item: item.name.casefold())[0] if len(matches) == 1 else None


def _find_task(root: Path, accepted: frozenset[str]) -> Path | None:
    return _find_child(root, accepted)


def _image_map(directory: Path, *, axis: str | None = None) -> tuple[dict[str, Path], int]:
    result: dict[str, Path] = {}
    duplicates = 0
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.casefold() not in _IMAGE_SUFFIXES:
            continue
        key = path.stem.casefold()
        if axis is not None:
            for suffix in _AXIS_SUFFIXES[axis]:
                if key.endswith(suffix):
                    key = key[: -len(suffix)]
                    break
        if key in result:
            duplicates += 1
        else:
            result[key] = path
    return result, duplicates


def _header_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _parse_int(value: str) -> int | None:
    try:
        return int(value.strip())
    except ValueError:
        return None


def _parse_grading_csv(path: Path) -> tuple[dict[str, tuple[int, int]], int, int]:
    rows: dict[str, tuple[int, int]] = {}
    malformed = 0
    duplicates = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            return rows, 1, 0
        names = {_header_key(name): name for name in reader.fieldnames}
        image_column = names.get("imagename") or names.get("imageid")
        dr_column = names.get("retinopathygrade") or names.get("drgrade")
        me_column = names.get("riskofmacularedema") or names.get("macularedemagrade")
        if image_column is None or dr_column is None or me_column is None:
            return rows, 1, 0
        for row in reader:
            image_value = row.get(image_column, "")
            dr_value = _parse_int(row.get(dr_column, ""))
            me_value = _parse_int(row.get(me_column, ""))
            key = Path(image_value).stem.casefold()
            if not key or dr_value is None or me_value is None:
                malformed += 1
            elif key in rows:
                duplicates += 1
            else:
                rows[key] = (dr_value, me_value)
    return rows, malformed, duplicates


def _find_csv(task_root: Path, split: str) -> Path | None:
    candidates = [
        path
        for path in task_root.rglob("*.csv")
        if split in _normalise_name(path.name)
    ]
    if len(candidates) != 1:
        return None
    return sorted(candidates, key=lambda item: item.as_posix().casefold())[0]


@dataclass(frozen=True, slots=True)
class _IDRiDGradingProvenance:
    status: str
    issues: tuple[str, ...]
    record_configured: bool
    record_available: bool
    official_source_declared: bool
    file_identity_declared: bool
    canonical_archive_available: bool
    sha256_match: bool | None


def _is_regular_file(path: Path, *, reject_symlink: bool = False) -> bool:
    try:
        return path.is_file() and (not reject_symlink or not path.is_symlink())
    except OSError:
        return False


def _component_absent_provenance(record: Path | None) -> _IDRiDGradingProvenance:
    return _IDRiDGradingProvenance(
        status="NOT_CHECKED_COMPONENT_ABSENT",
        issues=(),
        record_configured=record is not None,
        record_available=bool(record is not None and _is_regular_file(record)),
        official_source_declared=False,
        file_identity_declared=False,
        canonical_archive_available=False,
        sha256_match=None,
    )


def _mapping(value: object) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        return None
    return value


def _declared_sha256(file_record: Mapping[str, Any]) -> str | None:
    values: list[str] = []
    for key in ("sha256", "sha256_measured_local"):
        if key not in file_record:
            continue
        value = file_record[key]
        if not isinstance(value, str):
            return None
        normalised = value.strip().casefold()
        if not _SHA256_PATTERN.fullmatch(normalised):
            return None
        values.append(normalised)
    if not values or len(set(values)) != 1:
        return None
    return values[0]


def _verify_grading_provenance(
    root: Path, record_path: Path | None
) -> _IDRiDGradingProvenance:
    """Verify B against explicit repository evidence before reading extracted files."""

    archive = root / "archives" / _GRADING_ARCHIVE_NAME
    archive_available = _is_regular_file(archive, reject_symlink=True)
    if record_path is None:
        return _IDRiDGradingProvenance(
            status="REJECTED",
            issues=("IDRID_GRADING_PROVENANCE_RECORD_NOT_CONFIGURED",),
            record_configured=False,
            record_available=False,
            official_source_declared=False,
            file_identity_declared=False,
            canonical_archive_available=archive_available,
            sha256_match=None,
        )
    if not _is_regular_file(record_path):
        return _IDRiDGradingProvenance(
            status="REJECTED",
            issues=("IDRID_GRADING_PROVENANCE_RECORD_MISSING",),
            record_configured=True,
            record_available=False,
            official_source_declared=False,
            file_identity_declared=False,
            canonical_archive_available=archive_available,
            sha256_match=None,
        )

    try:
        loaded = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return _IDRiDGradingProvenance(
            status="REJECTED",
            issues=("IDRID_GRADING_PROVENANCE_RECORD_INVALID",),
            record_configured=True,
            record_available=True,
            official_source_declared=False,
            file_identity_declared=False,
            canonical_archive_available=archive_available,
            sha256_match=None,
        )

    record = _mapping(loaded)
    source = _mapping(record.get("source")) if record is not None else None
    file_record = _mapping(record.get("file")) if record is not None else None
    record_identity_valid = bool(
        record is not None
        and record.get("schema_version") == 1
        and record.get("dataset") == "idrid"
        and record.get("component") == _GRADING_COMPONENT
    )
    official_source = bool(
        source is not None
        and isinstance(source.get("record"), str)
        and source["record"].strip().casefold().rstrip("/") in _OFFICIAL_RECORD_IDENTIFIERS
        and source.get("distribution") == _OFFICIAL_DISTRIBUTION
        and source.get("acquisition") in _ALLOWED_ACQUISITION_METHODS
    )
    declared_digest = _declared_sha256(file_record) if file_record is not None else None
    file_identity = bool(
        file_record is not None
        and file_record.get("name") == _GRADING_ARCHIVE_NAME
        and declared_digest is not None
    )

    issues: list[str] = []
    if not record_identity_valid:
        issues.append("IDRID_GRADING_PROVENANCE_RECORD_INVALID")
    if not official_source:
        issues.append("IDRID_GRADING_PROVENANCE_SOURCE_NOT_OFFICIAL")
    if not file_identity:
        issues.append("IDRID_GRADING_PROVENANCE_FILE_DECLARATION_INVALID")
    if not archive_available:
        issues.append("IDRID_GRADING_PROVENANCE_CANONICAL_ARCHIVE_MISSING")

    digest_match: bool | None = None
    if not issues and declared_digest is not None:
        try:
            digest_match = sha256_file(archive) == declared_digest
        except OSError:
            issues.append("IDRID_GRADING_PROVENANCE_ARCHIVE_READ_ERROR")
        else:
            if not digest_match:
                issues.append("IDRID_GRADING_PROVENANCE_HASH_MISMATCH")

    return _IDRiDGradingProvenance(
        status="VERIFIED" if not issues and digest_match else "REJECTED",
        issues=tuple(sorted(set(issues))),
        record_configured=True,
        record_available=True,
        official_source_declared=official_source,
        file_identity_declared=file_identity,
        canonical_archive_available=archive_available,
        sha256_match=digest_match,
    )


@dataclass(frozen=True, slots=True)
class _IDRiDInspection:
    status: AdapterStatus
    issues: tuple[str, ...]
    grading_images: Mapping[str, Mapping[str, Path]]
    grades: Mapping[str, Mapping[str, tuple[int, int]]]
    segmentation_images: Mapping[str, Mapping[str, Path]]
    masks: Mapping[str, Mapping[str, Mapping[str, Path]]]
    grading_provenance: _IDRiDGradingProvenance


class IDRiDAdapter:
    """Audit the official IDRiD segmentation and grading task releases.

    The two task packages use independent split assignments and the public
    release does not provide a machine-readable cross-component identity
    table.  Grades are therefore joined only by an exact key inside the same
    named task split.  Zero-padding heuristics and visual-similarity matches
    are deliberately not used to assign clinical labels.
    """

    dataset_name = "idrid"

    def __init__(
        self,
        root: str | Path,
        *,
        grading_acquisition_record: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.grading_acquisition_record = (
            Path(grading_acquisition_record)
            if grading_acquisition_record is not None
            else None
        )
        self._grading_provenance_cache: _IDRiDGradingProvenance | None = None

    def _grading_provenance(self) -> _IDRiDGradingProvenance:
        if self._grading_provenance_cache is None:
            self._grading_provenance_cache = _verify_grading_provenance(
                self.root, self.grading_acquisition_record
            )
        return self._grading_provenance_cache

    def discover(self) -> DiscoveryResult:
        if not self.root.exists() or not self.root.is_dir():
            return unavailable_discovery(self.root)
        return DiscoveryResult(
            status=AdapterStatus.READY,
            root=self.root,
            files=None,
            message="IDRID_ROOT_AVAILABLE",
        )

    def _inspect(self) -> _IDRiDInspection:
        if not self.root.exists() or not self.root.is_dir():
            return _IDRiDInspection(
                status=AdapterStatus.DATA_NOT_AVAILABLE,
                issues=("IDRID_ROOT_UNAVAILABLE",),
                grading_images={},
                grades={},
                segmentation_images={},
                masks={},
                grading_provenance=_component_absent_provenance(
                    self.grading_acquisition_record
                ),
            )

        incomplete: list[str] = []
        invalid: list[str] = []
        unresolved: list[str] = []
        grading_images: dict[str, dict[str, Path]] = {}
        grades: dict[str, dict[str, tuple[int, int]]] = {}
        segmentation_images: dict[str, dict[str, Path]] = {}
        masks: dict[str, dict[str, dict[str, Path]]] = {}
        grading_provenance = _component_absent_provenance(
            self.grading_acquisition_record
        )
        grading_provenance_verified = False

        try:
            grading_task = _find_task(
                self.root, frozenset({"diseasegrading", "bdiseasegrading"})
            )
            segmentation_task = _find_task(
                self.root, frozenset({"segmentation", "asegmentation"})
            )
        except OSError:
            invalid.append("IDRID_TASK_DISCOVERY_READ_ERROR")
            grading_task = None
            segmentation_task = None

        if grading_task is None:
            incomplete.append("IDRID_GRADING_TASK_MISSING")
        else:
            grading_provenance = self._grading_provenance()
            grading_provenance_verified = grading_provenance.status == "VERIFIED"
            if not grading_provenance_verified:
                invalid.extend(grading_provenance.issues)
            else:
                original_images = _find_child(grading_task, frozenset({"originalimages"}))
                if original_images is None:
                    incomplete.append("IDRID_GRADING_IMAGES_ROOT_MISSING")
                else:
                    for split, accepted in _SPLIT_NAMES.items():
                        split_directory = _find_child(original_images, accepted)
                        label_file = _find_csv(grading_task, split)
                        if split_directory is None or label_file is None:
                            incomplete.append(f"IDRID_GRADING_SPLIT_MISSING:{split.upper()}")
                            continue
                        try:
                            split_images, duplicate_images = _image_map(split_directory)
                            split_grades, malformed, duplicate_labels = _parse_grading_csv(
                                label_file
                            )
                        except (OSError, UnicodeError):
                            invalid.append(f"IDRID_GRADING_READ_ERROR:{split.upper()}")
                            continue
                        grading_images[split] = split_images
                        grades[split] = split_grades
                        if duplicate_images:
                            invalid.append(
                                "IDRID_GRADING_IMAGE_KEY_DUPLICATES:"
                                f"{split.upper()}:{duplicate_images}"
                            )
                        if malformed:
                            invalid.append(
                                "IDRID_GRADING_LABEL_ROWS_INVALID:"
                                f"{split.upper()}:{malformed}"
                            )
                        if duplicate_labels:
                            invalid.append(
                                "IDRID_GRADING_LABEL_KEY_DUPLICATES:"
                                f"{split.upper()}:{duplicate_labels}"
                            )
                        missing = len(set(split_images) - set(split_grades))
                        orphan = len(set(split_grades) - set(split_images))
                        if missing or orphan:
                            invalid.append(
                                "IDRID_GRADING_IMAGE_LABEL_MISMATCH:"
                                f"{split.upper()}:missing={missing}:orphan={orphan}"
                            )

        if segmentation_task is None:
            incomplete.append("IDRID_SEGMENTATION_TASK_MISSING")
        else:
            original_images = _find_child(segmentation_task, frozenset({"originalimages"}))
            groundtruths = _find_child(
                segmentation_task, frozenset({"allsegmentationgroundtruths"})
            )
            if original_images is None or groundtruths is None:
                incomplete.append("IDRID_SEGMENTATION_STRUCTURE_INCOMPLETE")
            else:
                for split, accepted in _SPLIT_NAMES.items():
                    image_directory = _find_child(original_images, accepted)
                    groundtruth_split = _find_child(groundtruths, accepted)
                    # The official release may expose masks for training only.
                    # A split without a ground-truth directory is not converted
                    # to zero-valued lesion axes and is simply not sample-ready.
                    if groundtruth_split is None:
                        continue
                    if image_directory is None:
                        incomplete.append(
                            f"IDRID_SEGMENTATION_IMAGES_MISSING:{split.upper()}"
                        )
                        continue
                    try:
                        split_images, duplicate_images = _image_map(image_directory)
                    except OSError:
                        invalid.append(f"IDRID_SEGMENTATION_READ_ERROR:{split.upper()}")
                        continue
                    segmentation_images[split] = split_images
                    masks[split] = {}
                    if duplicate_images:
                        invalid.append(
                            "IDRID_SEGMENTATION_IMAGE_KEY_DUPLICATES:"
                            f"{split.upper()}:{duplicate_images}"
                        )
                    for axis, accepted_axis_names in _AXIS_NAMES.items():
                        axis_directory = _find_child(groundtruth_split, accepted_axis_names)
                        if axis_directory is None:
                            incomplete.append(
                                f"IDRID_SEGMENTATION_AXIS_MISSING:{split.upper()}:{axis.upper()}"
                            )
                            continue
                        try:
                            axis_masks, duplicate_masks = _image_map(
                                axis_directory, axis=axis
                            )
                        except OSError:
                            invalid.append(
                                f"IDRID_SEGMENTATION_AXIS_READ_ERROR:{split.upper()}:{axis.upper()}"
                            )
                            continue
                        masks[split][axis] = axis_masks
                        missing = len(set(split_images) - set(axis_masks))
                        orphan = len(set(axis_masks) - set(split_images))
                        if duplicate_masks:
                            invalid.append(
                                "IDRID_SEGMENTATION_MASK_KEY_DUPLICATES:"
                                f"{split.upper()}:{axis.upper()}:{duplicate_masks}"
                            )
                        if orphan:
                            invalid.append(
                                "IDRID_SEGMENTATION_IMAGE_MASK_MISMATCH:"
                                f"{split.upper()}:{axis.upper()}:missing=0:orphan={orphan}"
                            )
                        if missing:
                            unresolved.append(
                                "IDRID_SEGMENTATION_MASK_UNRESOLVED:"
                                f"{split.upper()}:{axis.upper()}:{missing}"
                            )
                        if split_images and not axis_masks:
                            incomplete.append(
                                "IDRID_SEGMENTATION_AXIS_LABELS_UNAVAILABLE:"
                                f"{split.upper()}:{axis.upper()}"
                            )
                if not segmentation_images:
                    incomplete.append("IDRID_SEGMENTATION_LABELS_UNAVAILABLE")

        if grading_provenance_verified:
            for split in ("train", "test"):
                segmentation_keys = set(segmentation_images.get(split, {}))
                missing_grades = len(segmentation_keys - set(grades.get(split, {})))
                if missing_grades:
                    unresolved.append(
                        "IDRID_SEGMENTATION_GRADE_UNRESOLVED:"
                        f"{split.upper()}:{missing_grades}"
                    )

        if invalid:
            status = AdapterStatus.INVALID
        elif incomplete:
            status = AdapterStatus.INCOMPLETE
        else:
            status = AdapterStatus.READY
        return _IDRiDInspection(
            status=status,
            issues=tuple(sorted(set((*invalid, *incomplete, *unresolved)))),
            grading_images=grading_images,
            grades=grades,
            segmentation_images=segmentation_images,
            masks=masks,
            grading_provenance=grading_provenance,
        )

    def validate(self) -> ValidationResult:
        inspection = self._safe_inspect()
        message = {
            AdapterStatus.READY: "IDRID_READY",
            AdapterStatus.INCOMPLETE: "IDRID_INCOMPLETE",
            AdapterStatus.INVALID: "IDRID_INVALID",
            AdapterStatus.DATA_NOT_AVAILABLE: "DATA_NOT_AVAILABLE",
        }[inspection.status]
        return ValidationResult(inspection.status, inspection.issues, message)

    def samples(self) -> tuple[DatasetSample, ...]:
        inspection = self._safe_inspect()
        require_ready(
            self.dataset_name,
            ValidationResult(inspection.status, inspection.issues, "IDRID_SAMPLE_CHECK"),
        )
        samples: list[DatasetSample] = []
        for split in ("train", "test"):
            split_images = inspection.segmentation_images.get(split, {})
            split_masks = inspection.masks.get(split, {})
            for source_key, image_path in sorted(split_images.items()):
                grade = inspection.grades.get(split, {}).get(source_key)
                dr_grade, me_grade = grade if grade is not None else (None, None)
                source_masks = {
                    axis: axis_masks[source_key]
                    for axis, axis_masks in split_masks.items()
                    if source_key in axis_masks
                }
                unresolved_axis_count = len(_AXIS_NAMES) - len(source_masks)
                samples.append(
                    DatasetSample(
                        sample_id=opaque_sample_id(
                            self.dataset_name, f"segmentation/{split}/{source_key}"
                        ),
                        image_path=image_path,
                        dataset_name=self.dataset_name,
                        official_split=split,
                        grade_fields={"dr_grade": dr_grade, "me_grade": me_grade},
                        lesion_masks=source_masks,
                        metadata={
                            "annotation_type": "pixel_mask",
                            "four_axis_complete": unresolved_axis_count == 0,
                            "axis_evidence_state": (
                                "COMPLETE"
                                if unresolved_axis_count == 0
                                else "PARTIAL_UNRESOLVED"
                            ),
                            "unresolved_axis_count": unresolved_axis_count,
                            "grade_available": grade is not None,
                            "grading_split": split if grade is not None else None,
                            "grade_linkage_state": (
                                "EXACT_SAME_SPLIT_KEY"
                                if grade is not None
                                else "UNRESOLVED_NO_OFFICIAL_CROSSWALK"
                            ),
                        },
                    )
                )
        return tuple(samples)

    def audit_facts(self) -> Mapping[str, object]:
        inspection = self._safe_inspect()
        grading_splits: dict[str, object] = {}
        for split in ("train", "test"):
            image_keys = set(inspection.grading_images.get(split, {}))
            split_grades = inspection.grades.get(split, {})
            grade_keys = set(split_grades)
            dr_distribution = Counter(values[0] for values in split_grades.values())
            me_distribution = Counter(values[1] for values in split_grades.values())
            grading_splits[split] = {
                "image_count": len(image_keys),
                "label_count": len(grade_keys),
                "matched_count": len(image_keys & grade_keys),
                "missing_label_count": len(image_keys - grade_keys),
                "orphan_label_count": len(grade_keys - image_keys),
                "dr_grade_distribution": {
                    str(value): dr_distribution[value] for value in sorted(dr_distribution)
                },
                "me_grade_distribution": {
                    str(value): me_distribution[value] for value in sorted(me_distribution)
                },
            }

        segmentation_splits: dict[str, object] = {}
        for split in ("train", "test"):
            if split not in inspection.segmentation_images:
                segmentation_splits[split] = {
                    "available": False,
                    "image_count": None,
                    "complete_four_axis_count": None,
                    "mask_counts": None,
                    "missing_mask_counts": None,
                    "missing_grade_count": None,
                }
                continue
            image_keys = set(inspection.segmentation_images[split])
            split_masks = inspection.masks.get(split, {})
            complete_keys = set(image_keys)
            for axis in _AXIS_NAMES:
                complete_keys &= set(split_masks.get(axis, {}))
            segmentation_splits[split] = {
                "available": True,
                "image_count": len(image_keys),
                "complete_four_axis_count": len(complete_keys),
                "mask_counts": {
                    axis: len(split_masks.get(axis, {})) for axis in _AXIS_NAMES
                },
                "missing_mask_counts": {
                    axis: len(image_keys - set(split_masks.get(axis, {})))
                    for axis in _AXIS_NAMES
                },
                "missing_grade_count": len(
                    image_keys - set(inspection.grades.get(split, {}))
                ),
            }

        return {
            "schema_version": 1,
            "dataset": self.dataset_name,
            "status": inspection.status.value,
            "issues": list(inspection.issues),
            "grading": {
                "provenance": {
                    "status": inspection.grading_provenance.status,
                    "record_configured": inspection.grading_provenance.record_configured,
                    "record_available": inspection.grading_provenance.record_available,
                    "official_source_declared": (
                        inspection.grading_provenance.official_source_declared
                    ),
                    "file_identity_declared": (
                        inspection.grading_provenance.file_identity_declared
                    ),
                    "canonical_archive_available": (
                        inspection.grading_provenance.canonical_archive_available
                    ),
                    "sha256_match": inspection.grading_provenance.sha256_match,
                },
                "splits": grading_splits,
                "total_image_count": sum(
                    len(inspection.grading_images.get(split, {}))
                    for split in ("train", "test")
                ),
                "total_label_count": sum(
                    len(inspection.grades.get(split, {})) for split in ("train", "test")
                ),
            },
            "segmentation": {
                "task_available": bool(inspection.segmentation_images),
                "absent_mask_semantics": "UNRESOLVED_NOT_NEGATIVE",
                "grade_linkage_semantics": (
                    "EXACT_SAME_TASK_SPLIT_KEY_ONLY_NO_VISUAL_INFERENCE"
                ),
                "splits": segmentation_splits,
                "total_complete_four_axis_count": (
                    sum(
                        int(facts["complete_four_axis_count"])
                        for facts in segmentation_splits.values()
                        if isinstance(facts, dict)
                        and facts["complete_four_axis_count"] is not None
                    )
                    if inspection.segmentation_images
                    else None
                ),
            },
        }

    def _safe_inspect(self) -> _IDRiDInspection:
        try:
            return self._inspect()
        except (OSError, UnicodeError):
            return _IDRiDInspection(
                status=AdapterStatus.INVALID,
                issues=("IDRID_ADAPTER_READ_ERROR",),
                grading_images={},
                grades={},
                segmentation_images={},
                masks={},
                grading_provenance=_component_absent_provenance(
                    self.grading_acquisition_record
                ),
            )


__all__ = ["IDRiDAdapter"]

"""Read-only adapter for the official DDR grading and lesion tasks."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from nts_med.data.adapters.base import (
    AdapterStatus,
    DatasetSample,
    DiscoveryResult,
    ValidationResult,
    opaque_sample_id,
    require_ready,
    unavailable_discovery,
)

_SPLITS = ("train", "valid", "test")
_AXIS_DIRECTORIES: Mapping[str, str] = {
    "ma": "MA",
    "hem": "HE",
    "ex": "EX",
    "cws": "SE",
}
_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
_TOKEN_SPLIT = re.compile(r"[,\s]+")


@dataclass(frozen=True, slots=True)
class _DDRInspection:
    status: AdapterStatus
    issues: tuple[str, ...]
    grading_images: Mapping[str, Mapping[str, Path]]
    grades: Mapping[str, Mapping[str, int]]
    segmentation_images: Mapping[str, Mapping[str, Path]]
    masks: Mapping[str, Mapping[str, Mapping[str, Path]]]
    grading_split_by_key: Mapping[str, str]
    grade_by_key: Mapping[str, int]


def _file_key(path: Path) -> str:
    return path.stem.casefold()


def _direct_file_map(directory: Path) -> tuple[dict[str, Path], int]:
    result: dict[str, Path] = {}
    duplicates = 0
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.casefold() not in _IMAGE_SUFFIXES:
            continue
        key = _file_key(path)
        if key in result:
            duplicates += 1
        else:
            result[key] = path
    return result, duplicates


def _parse_grade_file(path: Path) -> tuple[dict[str, int], int, int]:
    grades: dict[str, int] = {}
    malformed = 0
    duplicates = 0
    with path.open("r", encoding="utf-8-sig", errors="strict") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            tokens = [token for token in _TOKEN_SPLIT.split(line) if token]
            if len(tokens) < 2:
                malformed += 1
                continue
            try:
                grade = int(tokens[-1])
            except ValueError:
                # A single textual header is accepted; other non-integer rows
                # remain an aggregate parse error without exposing their text.
                if line_number == 1:
                    continue
                malformed += 1
                continue
            key = Path(tokens[0]).stem.casefold()
            if not key:
                malformed += 1
            elif key in grades:
                duplicates += 1
            else:
                grades[key] = grade
    return grades, malformed, duplicates


def _grade_distribution(grades: Mapping[str, int]) -> dict[str, int]:
    counts = Counter(grades.values())
    return {str(grade): counts[grade] for grade in sorted(counts)}


class DDRAdapter:
    """Validate and map the two independent official DDR task layouts.

    Lesion task splits and grading task splits are deliberately not assumed to
    coincide.  ``DatasetSample.official_split`` records the lesion split while
    the safe ``grading_split`` enum is retained in metadata when a grade match
    exists.
    """

    dataset_name = "ddr"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def discover(self) -> DiscoveryResult:
        """Report root availability without downloading or guessing content."""

        if not self.root.exists() or not self.root.is_dir():
            return unavailable_discovery(self.root)
        return DiscoveryResult(
            status=AdapterStatus.READY,
            root=self.root,
            files=None,
            message="DDR_ROOT_AVAILABLE",
        )

    def _inspect(self) -> _DDRInspection:
        if not self.root.exists() or not self.root.is_dir():
            return _DDRInspection(
                status=AdapterStatus.DATA_NOT_AVAILABLE,
                issues=("DDR_ROOT_UNAVAILABLE",),
                grading_images={},
                grades={},
                segmentation_images={},
                masks={},
                grading_split_by_key={},
                grade_by_key={},
            )

        incomplete: list[str] = []
        invalid: list[str] = []
        grading_images: dict[str, dict[str, Path]] = {}
        grades: dict[str, dict[str, int]] = {}
        segmentation_images: dict[str, dict[str, Path]] = {}
        masks: dict[str, dict[str, dict[str, Path]]] = {}

        grading_root = self.root / "DR_grading"
        lesion_root = self.root / "lesion_segmentation"
        if not grading_root.is_dir():
            incomplete.append("DDR_GRADING_TASK_MISSING")
        if not lesion_root.is_dir():
            incomplete.append("DDR_LESION_TASK_MISSING")

        if grading_root.is_dir():
            for split in _SPLITS:
                image_directory = grading_root / split
                label_file = grading_root / f"{split}.txt"
                if not image_directory.is_dir():
                    incomplete.append(f"DDR_GRADING_IMAGES_MISSING:{split.upper()}")
                    continue
                if not label_file.is_file():
                    incomplete.append(f"DDR_GRADING_LABELS_MISSING:{split.upper()}")
                    continue
                try:
                    image_map, duplicate_images = _direct_file_map(image_directory)
                    split_grades, malformed, duplicate_labels = _parse_grade_file(label_file)
                except (OSError, UnicodeError):
                    invalid.append(f"DDR_GRADING_READ_ERROR:{split.upper()}")
                    continue
                grading_images[split] = image_map
                grades[split] = split_grades
                if not image_map:
                    incomplete.append(f"DDR_GRADING_IMAGES_EMPTY:{split.upper()}")
                if not split_grades:
                    incomplete.append(f"DDR_GRADING_LABELS_EMPTY:{split.upper()}")
                if duplicate_images:
                    invalid.append(
                        f"DDR_GRADING_IMAGE_KEY_DUPLICATES:{split.upper()}:{duplicate_images}"
                    )
                if malformed:
                    invalid.append(f"DDR_GRADING_LABEL_ROWS_INVALID:{split.upper()}:{malformed}")
                if duplicate_labels:
                    invalid.append(
                        f"DDR_GRADING_LABEL_KEY_DUPLICATES:{split.upper()}:{duplicate_labels}"
                    )
                missing_labels = len(set(image_map) - set(split_grades))
                orphan_labels = len(set(split_grades) - set(image_map))
                if missing_labels or orphan_labels:
                    invalid.append(
                        "DDR_GRADING_IMAGE_LABEL_MISMATCH:"
                        f"{split.upper()}:missing={missing_labels}:orphan={orphan_labels}"
                    )

        if lesion_root.is_dir():
            for split in _SPLITS:
                split_root = lesion_root / split
                image_directory = split_root / "image"
                label_root = split_root / ("segmentation label" if split == "valid" else "label")
                if not image_directory.is_dir() or not label_root.is_dir():
                    incomplete.append(f"DDR_LESION_SPLIT_MISSING:{split.upper()}")
                    continue
                try:
                    image_map, duplicate_images = _direct_file_map(image_directory)
                except OSError:
                    invalid.append(f"DDR_LESION_READ_ERROR:{split.upper()}")
                    continue
                segmentation_images[split] = image_map
                masks[split] = {}
                if not image_map:
                    incomplete.append(f"DDR_LESION_IMAGES_EMPTY:{split.upper()}")
                if duplicate_images:
                    invalid.append(
                        f"DDR_LESION_IMAGE_KEY_DUPLICATES:{split.upper()}:{duplicate_images}"
                    )
                for axis, source_directory in _AXIS_DIRECTORIES.items():
                    axis_directory = label_root / source_directory
                    if not axis_directory.is_dir():
                        incomplete.append(
                            f"DDR_LESION_AXIS_MISSING:{split.upper()}:{axis.upper()}"
                        )
                        continue
                    try:
                        axis_masks, duplicate_masks = _direct_file_map(axis_directory)
                    except OSError:
                        invalid.append(
                            f"DDR_LESION_AXIS_READ_ERROR:{split.upper()}:{axis.upper()}"
                        )
                        continue
                    masks[split][axis] = axis_masks
                    missing_masks = len(set(image_map) - set(axis_masks))
                    orphan_masks = len(set(axis_masks) - set(image_map))
                    if duplicate_masks:
                        invalid.append(
                            "DDR_LESION_MASK_KEY_DUPLICATES:"
                            f"{split.upper()}:{axis.upper()}:{duplicate_masks}"
                        )
                    if missing_masks or orphan_masks:
                        invalid.append(
                            "DDR_LESION_IMAGE_MASK_MISMATCH:"
                            f"{split.upper()}:{axis.upper()}:"
                            f"missing={missing_masks}:orphan={orphan_masks}"
                        )

        grading_split_by_key: dict[str, str] = {}
        grade_by_key: dict[str, int] = {}
        ambiguous_grading_keys = 0
        for split in _SPLITS:
            for key, grade in grades.get(split, {}).items():
                if key in grade_by_key:
                    ambiguous_grading_keys += 1
                    continue
                grade_by_key[key] = grade
                grading_split_by_key[key] = split
        if ambiguous_grading_keys:
            invalid.append(f"DDR_CROSS_SPLIT_GRADING_KEY_DUPLICATES:{ambiguous_grading_keys}")

        segmentation_keys = {
            key for split_images in segmentation_images.values() for key in split_images
        }
        missing_segmentation_grades = len(segmentation_keys - set(grade_by_key))
        ready_warnings: list[str] = []
        if missing_segmentation_grades:
            # A lesion annotation remains usable for burden estimation when its
            # scalar grade is absent; debt analyses must exclude that row.
            ready_warnings.append(
                f"DDR_SEGMENTATION_GRADE_MISSING:{missing_segmentation_grades}"
            )
        task_split_conflicts = sum(
            grading_split_by_key.get(key) is not None
            and grading_split_by_key[key] != lesion_split
            for lesion_split, split_images in segmentation_images.items()
            for key in split_images
        )
        if task_split_conflicts:
            # The lesion task remains authoritative for analysis splits.  A
            # uniquely matched grading-task value is retained only as debt and
            # stratification metadata, with the task-split difference made
            # explicit.  It never changes the lesion split or four-axis relation.
            ready_warnings.append(
                f"DDR_GRADING_SPLIT_CONFLICT_RETAINED:{task_split_conflicts}"
            )

        if invalid:
            status = AdapterStatus.INVALID
        elif incomplete:
            status = AdapterStatus.INCOMPLETE
        else:
            status = AdapterStatus.READY
        issues = tuple(sorted(set((*invalid, *incomplete, *ready_warnings))))
        return _DDRInspection(
            status=status,
            issues=issues,
            grading_images=grading_images,
            grades=grades,
            segmentation_images=segmentation_images,
            masks=masks,
            grading_split_by_key=grading_split_by_key,
            grade_by_key=grade_by_key,
        )

    def validate(self) -> ValidationResult:
        """Validate known DDR task structure and aggregate key matching."""

        inspection = self._safe_inspect()
        message = {
            AdapterStatus.READY: "DDR_READY",
            AdapterStatus.INCOMPLETE: "DDR_INCOMPLETE",
            AdapterStatus.INVALID: "DDR_INVALID",
            AdapterStatus.DATA_NOT_AVAILABLE: "DATA_NOT_AVAILABLE",
        }[inspection.status]
        return ValidationResult(
            status=inspection.status,
            issues=inspection.issues,
            message=message,
        )

    def samples(self) -> tuple[DatasetSample, ...]:
        """Return all four-axis lesion samples, preserving lesion split labels."""

        inspection = self._safe_inspect()
        require_ready(
            self.dataset_name,
            ValidationResult(inspection.status, inspection.issues, "DDR_SAMPLE_CHECK"),
        )
        samples: list[DatasetSample] = []
        for lesion_split in _SPLITS:
            split_images = inspection.segmentation_images.get(lesion_split, {})
            split_masks = inspection.masks.get(lesion_split, {})
            for source_key, image_path in sorted(split_images.items()):
                grading_split = inspection.grading_split_by_key.get(source_key)
                grade = inspection.grade_by_key.get(source_key)
                grading_split_matches = grading_split == lesion_split
                if grade is None:
                    usable_grade = None
                    grade_match_status = "MISSING"
                elif grade == 5:
                    usable_grade = None
                    grade_match_status = "GRADE5_QUALITY_ONLY"
                elif grading_split_matches:
                    usable_grade = grade
                    grade_match_status = "MATCHED_SAME_TASK_SPLIT"
                else:
                    usable_grade = grade
                    grade_match_status = "RETAINED_CROSS_TASK_SPLIT"
                samples.append(
                    DatasetSample(
                        sample_id=opaque_sample_id(
                            self.dataset_name, f"lesion/{lesion_split}/{source_key}"
                        ),
                        image_path=image_path,
                        dataset_name=self.dataset_name,
                        official_split=lesion_split,
                        grade_fields={"dr_grade": usable_grade},
                        quality_fields={"ungradable_grade5": grade == 5},
                        lesion_masks={
                            axis: split_masks[axis][source_key]
                            for axis in _AXIS_DIRECTORIES
                        },
                        metadata={
                            "annotation_type": "pixel_mask",
                            "four_axis_complete": True,
                            "grade_available": usable_grade is not None,
                            "grade_source": (
                                "DDR_DR_GRADING_TASK" if grade is not None else None
                            ),
                            "grade_value": grade,
                            "analysis_split_source": "DDR_LESION_SEGMENTATION_TASK",
                            "grading_task_split": grading_split,
                            "grading_split_conflict": (
                                grade is not None and not grading_split_matches
                            ),
                            "grade_match_status": grade_match_status,
                        },
                    )
                )
        return tuple(samples)

    def audit_facts(self) -> Mapping[str, object]:
        """Return aggregate task counts without paths or source identifiers."""

        inspection = self._safe_inspect()
        grading_splits: dict[str, object] = {}
        for split in _SPLITS:
            image_keys = set(inspection.grading_images.get(split, {}))
            split_grades = inspection.grades.get(split, {})
            grade_keys = set(split_grades)
            grading_splits[split] = {
                "image_count": len(image_keys),
                "label_count": len(grade_keys),
                "matched_count": len(image_keys & grade_keys),
                "missing_label_count": len(image_keys - grade_keys),
                "orphan_label_count": len(grade_keys - image_keys),
                "grade_distribution": _grade_distribution(split_grades),
            }

        segmentation_splits: dict[str, object] = {}
        grading_cross_table: dict[str, object] = {}
        for lesion_split in _SPLITS:
            image_keys = set(inspection.segmentation_images.get(lesion_split, {}))
            split_masks = inspection.masks.get(lesion_split, {})
            mask_counts = {
                axis: len(split_masks.get(axis, {})) for axis in _AXIS_DIRECTORIES
            }
            missing_counts = {
                axis: len(image_keys - set(split_masks.get(axis, {})))
                for axis in _AXIS_DIRECTORIES
            }
            complete_keys = set(image_keys)
            for axis in _AXIS_DIRECTORIES:
                complete_keys &= set(split_masks.get(axis, {}))
            segmentation_splits[lesion_split] = {
                "image_count": len(image_keys),
                "complete_four_axis_count": len(complete_keys),
                "mask_counts": mask_counts,
                "missing_mask_counts": missing_counts,
                "missing_grade_count": len(image_keys - set(inspection.grade_by_key)),
                "grade_retained_count": sum(
                    inspection.grade_by_key.get(key) in range(5) for key in image_keys
                ),
                "grade5_quality_only_count": sum(
                    inspection.grade_by_key.get(key) == 5 for key in image_keys
                ),
                "grading_split_conflict_count": sum(
                    inspection.grading_split_by_key.get(key) is not None
                    and inspection.grading_split_by_key[key] != lesion_split
                    for key in image_keys
                ),
            }
            cross_counts = Counter(
                inspection.grading_split_by_key.get(key, "missing") for key in image_keys
            )
            grading_cross_table[lesion_split] = {
                split: cross_counts.get(split, 0) for split in (*_SPLITS, "missing")
            }

        return {
            "schema_version": 1,
            "dataset": self.dataset_name,
            "status": inspection.status.value,
            "issues": list(inspection.issues),
            "grading": {
                "splits": grading_splits,
                "total_image_count": sum(
                    len(inspection.grading_images.get(split, {})) for split in _SPLITS
                ),
                "total_label_count": sum(
                    len(inspection.grades.get(split, {})) for split in _SPLITS
                ),
            },
            "segmentation": {
                "splits": segmentation_splits,
                "total_image_count": sum(
                    len(inspection.segmentation_images.get(split, {})) for split in _SPLITS
                ),
                "grading_split_cross_counts": grading_cross_table,
            },
        }

    def _safe_inspect(self) -> _DDRInspection:
        try:
            return self._inspect()
        except (OSError, UnicodeError):
            return _DDRInspection(
                status=AdapterStatus.INVALID,
                issues=("DDR_ADAPTER_READ_ERROR",),
                grading_images={},
                grades={},
                segmentation_images={},
                masks={},
                grading_split_by_key={},
                grade_by_key={},
            )


__all__ = ["DDRAdapter"]

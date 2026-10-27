"""Deterministic grouped split assignment and leakage validation.

Grouping is always explicit: callers select patient, eye, or image-instance
grouping and missing identifiers are errors rather than silent fallbacks.
Derived group identifiers are opaque SHA256 values so patient/eye identifiers
do not need to appear in split manifests or validation summaries.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

SPLIT_NAMES: Final[tuple[str, ...]] = (
    "train",
    "validation",
    "calibration",
    "test",
)


class GroupingLevel(StrEnum):
    """Leakage-control unit selected for one dataset."""

    PATIENT = "patient"
    EYE = "eye"
    INSTANCE = "instance"


class SplitValidationStatus(StrEnum):
    """Outcome of checking whether groups cross existing splits."""

    CLEAN = "CLEAN"
    GROUP_LEAKAGE = "GROUP_LEAKAGE"


@dataclass(frozen=True, slots=True)
class SplitRatios:
    """Four-way split fractions; defaults follow the preregistered 70/10/10/10."""

    train: float = 0.70
    validation: float = 0.10
    calibration: float = 0.10
    test: float = 0.10

    def __post_init__(self) -> None:
        values = self.as_tuple()
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise TypeError("split ratios must be real numbers")
        converted = tuple(float(value) for value in values)
        if any(not math.isfinite(value) or value < 0.0 for value in converted):
            raise ValueError("split ratios must be finite and non-negative")
        if not math.isclose(sum(converted), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("split ratios must sum to one")
        for name, value in zip(SPLIT_NAMES, converted, strict=True):
            object.__setattr__(self, name, value)

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return fractions in canonical train/validation/calibration/test order."""

        return (self.train, self.validation, self.calibration, self.test)


DEFAULT_SPLIT_RATIOS: Final[SplitRatios] = SplitRatios()


@dataclass(frozen=True, slots=True)
class InstanceIdentifiers:
    """Identifiers needed to choose an explicit grouping unit.

    ``instance_id`` must already be a non-patient, output-safe identifier when
    assignments will be serialized.  Patient and eye identifiers are used only
    to derive opaque group hashes and are not copied into outputs.
    """

    instance_id: str
    patient_id: str | None = None
    eye_id: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.instance_id, name="instance_id")
        if self.patient_id is not None:
            _require_identifier(self.patient_id, name="patient_id")
        if self.eye_id is not None:
            _require_identifier(self.eye_id, name="eye_id")


@dataclass(frozen=True, order=True, slots=True)
class GroupedInstance:
    """One output-safe instance identifier linked to an opaque grouping key."""

    instance_id: str
    group_id: str
    grouping_level: GroupingLevel


@dataclass(frozen=True, order=True, slots=True)
class SplitAssignment:
    """One deterministic split assignment."""

    instance_id: str
    split: str
    group_id: str
    grouping_level: GroupingLevel


@dataclass(frozen=True, slots=True)
class GroupLeakage:
    """An opaque group occurring in more than one existing split."""

    group_id: str
    splits: tuple[str, ...]
    instance_count: int


@dataclass(frozen=True, slots=True)
class SplitValidation:
    """Aggregate group-leakage validation without raw patient/eye identifiers."""

    status: SplitValidationStatus
    grouping_level: GroupingLevel
    instance_count: int
    group_count: int
    split_counts: tuple[tuple[str, int], ...]
    leakages: tuple[GroupLeakage, ...] = ()


def _require_identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"{name} must not contain a null byte")
    return value


def _require_grouping_level(value: object) -> GroupingLevel:
    if not isinstance(value, GroupingLevel):
        raise TypeError("grouping_level must be a GroupingLevel")
    return value


def _opaque_group_id(namespace: str, level: GroupingLevel, parts: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"nts-med-split-group-v1\0")
    for value in (namespace, level.value, *parts):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return f"grp_{digest.hexdigest()}"


def group_instances(
    instances: Iterable[InstanceIdentifiers],
    *,
    grouping_level: GroupingLevel,
    namespace: str,
) -> tuple[GroupedInstance, ...]:
    """Derive deterministic opaque groups at one explicitly selected level."""

    level = _require_grouping_level(grouping_level)
    namespace_value = _require_identifier(namespace, name="namespace")
    materialized = tuple(sorted(instances, key=lambda item: item.instance_id))
    identifiers = [item.instance_id for item in materialized]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("instances must have unique instance_id values")

    if level is GroupingLevel.PATIENT and any(
        item.patient_id is None for item in materialized
    ):
        raise ValueError("patient grouping requires patient_id for every instance")
    if level is GroupingLevel.EYE:
        if any(item.eye_id is None for item in materialized):
            raise ValueError("eye grouping requires eye_id for every instance")
        patients_present = [item.patient_id is not None for item in materialized]
        if any(patients_present) and not all(patients_present):
            raise ValueError(
                "eye grouping requires patient_id for every instance or for none of them"
            )

    grouped: list[GroupedInstance] = []
    for item in materialized:
        parts: tuple[str, ...]
        if level is GroupingLevel.PATIENT:
            assert item.patient_id is not None
            parts = (item.patient_id,)
        elif level is GroupingLevel.EYE:
            assert item.eye_id is not None
            parts = (
                (item.patient_id, item.eye_id)
                if item.patient_id is not None
                else (item.eye_id,)
            )
        else:
            parts = (item.instance_id,)
        grouped.append(
            GroupedInstance(
                instance_id=item.instance_id,
                group_id=_opaque_group_id(namespace_value, level, parts),
                grouping_level=level,
            )
        )
    return tuple(grouped)


def allocation_counts(
    group_count: int,
    ratios: SplitRatios = DEFAULT_SPLIT_RATIOS,
) -> tuple[int, int, int, int]:
    """Allocate whole groups by deterministic largest remainder."""

    if isinstance(group_count, bool) or not isinstance(group_count, int):
        raise TypeError("group_count must be an integer")
    if group_count < 0:
        raise ValueError("group_count must be non-negative")
    if not isinstance(ratios, SplitRatios):
        raise TypeError("ratios must be a SplitRatios")

    ideals = tuple(group_count * ratio for ratio in ratios.as_tuple())
    counts = [math.floor(value) for value in ideals]
    remainder = group_count - sum(counts)
    priority = sorted(
        range(len(SPLIT_NAMES)),
        key=lambda index: (-(ideals[index] - counts[index]), index),
    )
    for index in priority[:remainder]:
        counts[index] += 1
    return tuple(counts)  # type: ignore[return-value]


def _seeded_group_order(group_ids: Iterable[str], seed: int) -> tuple[str, ...]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    def ordering_key(group_id: str) -> tuple[bytes, str]:
        digest = hashlib.sha256()
        digest.update(b"nts-med-split-order-v1\0")
        digest.update(str(seed).encode("ascii"))
        digest.update(b"\0")
        digest.update(group_id.encode("ascii"))
        return digest.digest(), group_id

    return tuple(sorted(group_ids, key=ordering_key))


def assign_deterministic_splits(
    instances: Iterable[InstanceIdentifiers],
    *,
    grouping_level: GroupingLevel,
    namespace: str,
    seed: int,
    ratios: SplitRatios = DEFAULT_SPLIT_RATIOS,
) -> tuple[SplitAssignment, ...]:
    """Assign whole patient/eye/instance groups to four deterministic splits.

    Ratios are applied to unique group counts.  The assignment is invariant to
    caller input order and does not inspect labels, grades, or image content.
    """

    grouped = group_instances(
        instances,
        grouping_level=grouping_level,
        namespace=namespace,
    )
    group_ids = tuple(sorted({item.group_id for item in grouped}))
    ordered_groups = _seeded_group_order(group_ids, seed)
    counts = allocation_counts(len(group_ids), ratios)

    split_by_group: dict[str, str] = {}
    offset = 0
    for split, count in zip(SPLIT_NAMES, counts, strict=True):
        for group_id in ordered_groups[offset : offset + count]:
            split_by_group[group_id] = split
        offset += count

    return tuple(
        SplitAssignment(
            instance_id=item.instance_id,
            split=split_by_group[item.group_id],
            group_id=item.group_id,
            grouping_level=item.grouping_level,
        )
        for item in grouped
    )


def validate_group_leakage(
    assignments: Iterable[SplitAssignment],
) -> SplitValidation:
    """Validate that no opaque group occurs in more than one existing split."""

    materialized = tuple(sorted(assignments, key=lambda item: item.instance_id))
    if not materialized:
        raise ValueError("at least one split assignment is required")
    identifiers = [item.instance_id for item in materialized]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("split assignments must have unique instance_id values")
    levels = {item.grouping_level for item in materialized}
    if len(levels) != 1:
        raise ValueError("split assignments must use one grouping level")
    for item in materialized:
        _require_identifier(item.instance_id, name="instance_id")
        _require_identifier(item.group_id, name="group_id")
        _require_identifier(item.split, name="split")

    splits_by_group: defaultdict[str, set[str]] = defaultdict(set)
    count_by_group: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for item in materialized:
        splits_by_group[item.group_id].add(item.split)
        count_by_group[item.group_id] += 1
        split_counts[item.split] += 1

    leakages = tuple(
        GroupLeakage(
            group_id=group_id,
            splits=tuple(sorted(splits)),
            instance_count=count_by_group[group_id],
        )
        for group_id, splits in sorted(splits_by_group.items())
        if len(splits) > 1
    )
    level = next(iter(levels))
    return SplitValidation(
        status=(
            SplitValidationStatus.GROUP_LEAKAGE
            if leakages
            else SplitValidationStatus.CLEAN
        ),
        grouping_level=level,
        instance_count=len(materialized),
        group_count=len(splits_by_group),
        split_counts=tuple(sorted(split_counts.items())),
        leakages=leakages,
    )


def validate_existing_splits(
    instances: Iterable[InstanceIdentifiers],
    split_by_instance: Mapping[str, str],
    *,
    grouping_level: GroupingLevel,
    namespace: str,
) -> SplitValidation:
    """Validate official/existing split labels under an explicit group policy."""

    grouped = group_instances(
        instances,
        grouping_level=grouping_level,
        namespace=namespace,
    )
    identifiers = {item.instance_id for item in grouped}
    if set(split_by_instance) != identifiers:
        missing = sorted(identifiers - set(split_by_instance))
        unexpected = sorted(set(split_by_instance) - identifiers)
        raise ValueError(
            "split mapping must exactly cover instances "
            f"(missing={missing}, unexpected={unexpected})"
        )
    assignments = tuple(
        SplitAssignment(
            instance_id=item.instance_id,
            split=split_by_instance[item.instance_id],
            group_id=item.group_id,
            grouping_level=item.grouping_level,
        )
        for item in grouped
    )
    return validate_group_leakage(assignments)


__all__ = [
    "DEFAULT_SPLIT_RATIOS",
    "SPLIT_NAMES",
    "GroupLeakage",
    "GroupedInstance",
    "GroupingLevel",
    "InstanceIdentifiers",
    "SplitAssignment",
    "SplitRatios",
    "SplitValidation",
    "SplitValidationStatus",
    "allocation_counts",
    "assign_deterministic_splits",
    "group_instances",
    "validate_existing_splits",
    "validate_group_leakage",
]

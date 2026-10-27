"""Deterministic relation algebra for interval-valued disease burdens.

The rules in this module are deliberately independent of any model.  Larger
burden values mean worse disease on every axis.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class Relation(StrEnum):
    """The five pre-registered instance-level relations."""

    LESS = "LESS"
    GREATER = "GREATER"
    EQUIVALENT = "EQUIVALENT"
    STRUCTURAL_INCOMPARABLE = "STRUCTURAL_INCOMPARABLE"
    UNRESOLVED = "UNRESOLVED"


class UnresolvedReason(StrEnum):
    """Predeclared epistemic reasons that never receive transitive inference."""

    MISSING_AXIS = "UNRESOLVED_MISSING_AXIS"
    INVALID_EVIDENCE = "UNRESOLVED_INVALID_EVIDENCE"
    QUALITY = "UNRESOLVED_QUALITY"
    FOV = "UNRESOLVED_FOV"
    MARGIN_GAP = "UNRESOLVED_MARGIN_GAP"
    INTERVAL_OVERLAP = "UNRESOLVED_INTERVAL_OVERLAP"
    OTHER_PREDECLARED = "UNRESOLVED_OTHER_PREDECLARED"


def _finite_float(value: object, *, name: str) -> float:
    """Return ``value`` as a finite float, rejecting booleans explicitly."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite, got {converted!r}")
    return converted


@dataclass(frozen=True, slots=True)
class Interval:
    """A validated, immutable closed interval."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        lower = _finite_float(self.lower, name="lower")
        upper = _finite_float(self.upper, name="upper")
        if lower > upper:
            raise ValueError(f"lower must be <= upper, got [{lower}, {upper}]")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def center(self) -> float:
        """Return the overflow-safe midpoint of the interval."""

        return self.lower / 2.0 + self.upper / 2.0

    @property
    def width(self) -> float:
        """Return the interval width."""

        return self.upper - self.lower


def _validated_threshold_mapping(
    values: Mapping[str, float], *, name: str
) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")

    validated: dict[str, float] = {}
    for axis, value in values.items():
        if not isinstance(axis, str) or not axis:
            raise ValueError(f"{name} axis names must be non-empty strings")
        converted = _finite_float(value, name=f"{name}[{axis!r}]")
        if converted < 0.0:
            raise ValueError(f"{name}[{axis!r}] must be non-negative")
        validated[axis] = converted
    return MappingProxyType(dict(sorted(validated.items())))


@dataclass(frozen=True, slots=True)
class RelationConfig:
    """Thresholds used by the fixed relation rules.

    All three threshold mappings must name the same non-empty axis set.
    ``numeric_tolerance`` is solely for floating-point comparisons; it does not
    replace a scientifically calibrated ``delta`` or ``epsilon``.
    """

    delta: Mapping[str, float]
    epsilon: Mapping[str, float]
    max_width: Mapping[str, float]
    numeric_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        delta = _validated_threshold_mapping(self.delta, name="delta")
        epsilon = _validated_threshold_mapping(self.epsilon, name="epsilon")
        max_width = _validated_threshold_mapping(self.max_width, name="max_width")
        tolerance = _finite_float(self.numeric_tolerance, name="numeric_tolerance")
        if tolerance < 0.0:
            raise ValueError("numeric_tolerance must be non-negative")

        axes = set(delta)
        if not axes:
            raise ValueError("relation configuration must contain at least one axis")
        if set(epsilon) != axes or set(max_width) != axes:
            raise ValueError(
                "delta, epsilon, and max_width must use identical axis sets"
            )

        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "max_width", max_width)
        object.__setattr__(self, "numeric_tolerance", tolerance)

    @property
    def axes(self) -> tuple[str, ...]:
        """Return axes in deterministic order."""

        return tuple(self.delta)


@dataclass(frozen=True, slots=True)
class RelationEvidence:
    """Auditable evidence produced by :func:`infer_relation_with_evidence`."""

    relation: Relation
    left_worse_axes: tuple[str, ...]
    right_worse_axes: tuple[str, ...]
    dominance_axes: tuple[str, ...]
    equivalent_axes: tuple[str, ...]
    unresolved_axes: tuple[str, ...]
    minimum_margin: float | None
    reason: str
    unresolved_reason: UnresolvedReason | None = None

    @property
    def trigger_axes(self) -> tuple[str, ...]:
        """Return the axes that directly triggered the final relation."""

        if self.relation is Relation.STRUCTURAL_INCOMPARABLE:
            return tuple(sorted((*self.left_worse_axes, *self.right_worse_axes)))
        if self.relation in {Relation.LESS, Relation.GREATER}:
            return self.dominance_axes
        if self.relation is Relation.EQUIVALENT:
            return self.equivalent_axes
        return self.unresolved_axes


def unresolved_relation_evidence(
    unresolved_reason: UnresolvedReason,
    *,
    unresolved_axes: Sequence[str] = (),
) -> RelationEvidence:
    """Create fail-closed evidence when relation inference cannot be attempted."""

    if not isinstance(unresolved_reason, UnresolvedReason):
        raise TypeError("unresolved_reason must be an UnresolvedReason")
    axes = tuple(sorted(unresolved_axes))
    if any(not isinstance(axis, str) or not axis for axis in axes):
        raise ValueError("unresolved axis names must be non-empty strings")
    if len(set(axes)) != len(axes):
        raise ValueError("unresolved axis names must be unique")
    return RelationEvidence(
        relation=Relation.UNRESOLVED,
        left_worse_axes=(),
        right_worse_axes=(),
        dominance_axes=(),
        equivalent_axes=(),
        unresolved_axes=axes,
        minimum_margin=None,
        reason=unresolved_reason.value,
        unresolved_reason=unresolved_reason,
    )


def _validated_intervals(
    values: Mapping[str, Interval], *, name: str
) -> Mapping[str, Interval]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    validated: dict[str, Interval] = {}
    for axis, interval in values.items():
        if not isinstance(axis, str) or not axis:
            raise ValueError(f"{name} axis names must be non-empty strings")
        if not isinstance(interval, Interval):
            raise TypeError(f"{name}[{axis!r}] must be an Interval")
        validated[axis] = interval
    if not validated:
        raise ValueError(f"{name} must contain at least one axis")
    return MappingProxyType(validated)


def _normalise_zero(value: float, tolerance: float) -> float:
    return 0.0 if abs(value) <= tolerance else value


def infer_relation(
    left: Mapping[str, Interval],
    right: Mapping[str, Interval],
    config: RelationConfig,
) -> Relation:
    """Infer one relation using the pre-registered precedence order."""

    return infer_relation_with_evidence(left, right, config).relation


def infer_relation_with_evidence(
    left: Mapping[str, Interval],
    right: Mapping[str, Interval],
    config: RelationConfig,
) -> RelationEvidence:
    """Infer a relation and return the axis-level evidence used to decide it.

    Precedence is dominance, structural incomparability, equivalence, then
    unresolvedness.  A dominance threshold is inclusive: a gap exactly equal
    to ``delta`` passes.  At least one axis must nevertheless have a genuine
    positive endpoint gap, which prevents identical point intervals from being
    labelled as ordered when ``delta`` is zero.
    """

    if not isinstance(config, RelationConfig):
        raise TypeError("config must be a RelationConfig")
    left_intervals = _validated_intervals(left, name="left")
    right_intervals = _validated_intervals(right, name="right")

    left_axes = set(left_intervals)
    right_axes = set(right_intervals)
    config_axes = set(config.axes)
    if left_axes != right_axes:
        missing_left = sorted(right_axes - left_axes)
        missing_right = sorted(left_axes - right_axes)
        raise ValueError(
            "left and right must use identical axis sets "
            f"(missing from left={missing_left}, missing from right={missing_right})"
        )
    if left_axes != config_axes:
        missing_values = sorted(config_axes - left_axes)
        unexpected_values = sorted(left_axes - config_axes)
        raise ValueError(
            "interval axes must match the relation configuration "
            f"(missing={missing_values}, unexpected={unexpected_values})"
        )

    axes = tuple(sorted(left_axes))
    tolerance = config.numeric_tolerance

    # Positive values indicate separation beyond the configured delta.
    left_less_margins = {
        axis: right_intervals[axis].lower
        - left_intervals[axis].upper
        - config.delta[axis]
        for axis in axes
    }
    left_greater_margins = {
        axis: left_intervals[axis].lower
        - right_intervals[axis].upper
        - config.delta[axis]
        for axis in axes
    }

    left_positive_gap_axes = tuple(
        axis
        for axis in axes
        if right_intervals[axis].lower - left_intervals[axis].upper > tolerance
    )
    right_positive_gap_axes = tuple(
        axis
        for axis in axes
        if left_intervals[axis].lower - right_intervals[axis].upper > tolerance
    )

    left_dominates_as_less = all(
        margin >= -tolerance for margin in left_less_margins.values()
    ) and bool(left_positive_gap_axes)
    left_dominates_as_greater = all(
        margin >= -tolerance for margin in left_greater_margins.values()
    ) and bool(right_positive_gap_axes)

    left_worse_axes = tuple(
        axis for axis in axes if left_greater_margins[axis] > tolerance
    )
    right_worse_axes = tuple(
        axis for axis in axes if left_less_margins[axis] > tolerance
    )

    equivalent_axes = tuple(
        axis
        for axis in axes
        if abs(left_intervals[axis].center - right_intervals[axis].center)
        <= config.epsilon[axis] + tolerance
        and left_intervals[axis].width <= config.max_width[axis] + tolerance
        and right_intervals[axis].width <= config.max_width[axis] + tolerance
    )

    if left_dominates_as_less:
        margin = min(left_less_margins.values())
        return RelationEvidence(
            relation=Relation.LESS,
            left_worse_axes=left_worse_axes,
            right_worse_axes=right_worse_axes,
            dominance_axes=left_positive_gap_axes,
            equivalent_axes=equivalent_axes,
            unresolved_axes=(),
            minimum_margin=_normalise_zero(margin, tolerance),
            reason="left is no worse on every axis and is strictly lighter on at least one axis",
        )

    if left_dominates_as_greater:
        margin = min(left_greater_margins.values())
        return RelationEvidence(
            relation=Relation.GREATER,
            left_worse_axes=left_worse_axes,
            right_worse_axes=right_worse_axes,
            dominance_axes=right_positive_gap_axes,
            equivalent_axes=equivalent_axes,
            unresolved_axes=(),
            minimum_margin=_normalise_zero(margin, tolerance),
            reason="left is no lighter on every axis and is strictly worse on at least one axis",
        )

    if left_worse_axes and right_worse_axes:
        crossing_margins = tuple(
            left_greater_margins[axis] for axis in left_worse_axes
        ) + tuple(left_less_margins[axis] for axis in right_worse_axes)
        return RelationEvidence(
            relation=Relation.STRUCTURAL_INCOMPARABLE,
            left_worse_axes=left_worse_axes,
            right_worse_axes=right_worse_axes,
            dominance_axes=(),
            equivalent_axes=equivalent_axes,
            unresolved_axes=(),
            minimum_margin=min(crossing_margins),
            reason="different axes provide high-confidence evidence in opposite directions",
        )

    if len(equivalent_axes) == len(axes):
        equivalence_slacks = tuple(
            slack
            for axis in axes
            for slack in (
                config.epsilon[axis]
                - abs(left_intervals[axis].center - right_intervals[axis].center),
                config.max_width[axis] - left_intervals[axis].width,
                config.max_width[axis] - right_intervals[axis].width,
            )
        )
        return RelationEvidence(
            relation=Relation.EQUIVALENT,
            left_worse_axes=left_worse_axes,
            right_worse_axes=right_worse_axes,
            dominance_axes=(),
            equivalent_axes=equivalent_axes,
            unresolved_axes=(),
            minimum_margin=_normalise_zero(min(equivalence_slacks), tolerance),
            reason="all axes meet the centre-difference and interval-width equivalence bounds",
        )

    unresolved_axes = tuple(axis for axis in axes if axis not in equivalent_axes)
    unresolved_reason = (
        UnresolvedReason.INTERVAL_OVERLAP
        if any(
            left_intervals[axis].width > tolerance
            or right_intervals[axis].width > tolerance
            for axis in unresolved_axes
        )
        else UnresolvedReason.MARGIN_GAP
    )
    return RelationEvidence(
        relation=Relation.UNRESOLVED,
        left_worse_axes=left_worse_axes,
        right_worse_axes=right_worse_axes,
        dominance_axes=(),
        equivalent_axes=equivalent_axes,
        unresolved_axes=unresolved_axes,
        minimum_margin=None,
        reason="available intervals establish neither dominance, crossing, nor equivalence",
        unresolved_reason=unresolved_reason,
    )


__all__ = [
    "Interval",
    "Relation",
    "RelationConfig",
    "RelationEvidence",
    "UnresolvedReason",
    "infer_relation",
    "infer_relation_with_evidence",
    "unresolved_relation_evidence",
]

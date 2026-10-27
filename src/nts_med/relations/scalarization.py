"""Pure functions for auditing information debt introduced by scalar grades."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

from nts_med.relations.algebra import Relation


class MetricStatus(StrEnum):
    """Status of a scalarization audit rate."""

    OK = "OK"
    EMPTY_DENOMINATOR = "EMPTY_DENOMINATOR"


@dataclass(frozen=True, slots=True)
class RateResult:
    """A rate together with enough information to audit its denominator."""

    value: float | None
    numerator: int
    denominator: int
    status: MetricStatus


# Descriptive aliases make the return contract easy to discover for callers.
ScalarizationResult = RateResult
DebtResult = RateResult

GradeT = TypeVar("GradeT")


def debt_grade_indices(
    grades: Sequence[int | None],
    grading_split_conflicts: Sequence[bool],
    *,
    include_conflicts: bool,
) -> tuple[int, ...]:
    """Select grade-bearing rows for main or conflict-exclusion sensitivity debt.

    Grade values must be DDR severity grades 0--4.  Grade 5 is a quality state
    and must already be represented as missing by the adapter.
    """

    if not isinstance(include_conflicts, bool):
        raise TypeError("include_conflicts must be a bool")
    _require_equal_lengths(grades=grades, grading_split_conflicts=grading_split_conflicts)
    selected: list[int] = []
    for index, (grade, conflict) in enumerate(
        zip(grades, grading_split_conflicts, strict=True)
    ):
        if not isinstance(conflict, bool):
            raise TypeError("grading_split_conflicts must contain bool values")
        if grade is None:
            continue
        if isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= 4:
            raise ValueError("debt grades must be integers in 0--4 or None")
        if include_conflicts or not conflict:
            selected.append(index)
    return tuple(selected)


def _rate(numerator: int, denominator: int) -> RateResult:
    if denominator == 0:
        return RateResult(
            value=None,
            numerator=0,
            denominator=0,
            status=MetricStatus.EMPTY_DENOMINATOR,
        )
    return RateResult(
        value=numerator / denominator,
        numerator=numerator,
        denominator=denominator,
        status=MetricStatus.OK,
    )


def _require_equal_lengths(**sequences: Sequence[object]) -> int:
    lengths = {name: len(sequence) for name, sequence in sequences.items()}
    if len(set(lengths.values())) > 1:
        rendered = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"input sequences must have equal lengths ({rendered})")
    return next(iter(lengths.values()), 0)


def _validate_relations(relations: Sequence[Relation]) -> None:
    if isinstance(relations, (str, bytes)):
        raise TypeError("relations must be a sequence of Relation values")
    for index, relation in enumerate(relations):
        if not isinstance(relation, Relation):
            raise TypeError(f"relations[{index}] must be a Relation")


def _validate_grades(grades: Sequence[GradeT], *, name: str) -> None:
    if isinstance(grades, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of grade values")
    for index, grade in enumerate(grades):
        if grade is None:
            raise ValueError(f"{name}[{index}] must not be None")
        if isinstance(grade, float) and not math.isfinite(grade):
            raise ValueError(f"{name}[{index}] must be finite")


def forced_order_debt(
    relations: Sequence[Relation],
    left_grades: Sequence[GradeT],
    right_grades: Sequence[GradeT],
) -> RateResult:
    """Return ``P(STRUCTURAL_INCOMPARABLE | left_grade != right_grade)``."""

    _require_equal_lengths(
        relations=relations,
        left_grades=left_grades,
        right_grades=right_grades,
    )
    _validate_relations(relations)
    _validate_grades(left_grades, name="left_grades")
    _validate_grades(right_grades, name="right_grades")

    eligible = tuple(
        index
        for index, (left_grade, right_grade) in enumerate(
            zip(left_grades, right_grades, strict=True)
        )
        if left_grade != right_grade
    )
    numerator = sum(
        relations[index] is Relation.STRUCTURAL_INCOMPARABLE for index in eligible
    )
    return _rate(numerator, len(eligible))


def within_grade_compression_debt(
    relations: Sequence[Relation],
    left_grades: Sequence[GradeT],
    right_grades: Sequence[GradeT],
    *,
    include_dominance: bool = False,
) -> RateResult:
    """Return WCD, or WCD+ when ``include_dominance`` is true.

    WCD counts structurally incomparable pairs within the same grade.  WCD+
    additionally counts explicit ``LESS`` and ``GREATER`` pairs.
    """

    if not isinstance(include_dominance, bool):
        raise TypeError("include_dominance must be a bool")
    _require_equal_lengths(
        relations=relations,
        left_grades=left_grades,
        right_grades=right_grades,
    )
    _validate_relations(relations)
    _validate_grades(left_grades, name="left_grades")
    _validate_grades(right_grades, name="right_grades")

    counted_relations = {Relation.STRUCTURAL_INCOMPARABLE}
    if include_dominance:
        counted_relations.update({Relation.LESS, Relation.GREATER})
    eligible = tuple(
        index
        for index, (left_grade, right_grade) in enumerate(
            zip(left_grades, right_grades, strict=True)
        )
        if left_grade == right_grade
    )
    numerator = sum(relations[index] in counted_relations for index in eligible)
    return _rate(numerator, len(eligible))


def _validated_scores(scores: Sequence[float], *, name: str) -> tuple[float, ...]:
    if isinstance(scores, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of numeric scores")
    validated: list[float] = []
    for index, score in enumerate(scores):
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError(f"{name}[{index}] must be a real number")
        converted = float(score)
        if not math.isfinite(converted):
            raise ValueError(f"{name}[{index}] must be finite")
        validated.append(converted)
    return tuple(validated)


def _sign(value: float, tolerance: float) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def policy_reversal_rate(
    policy_a_left_scores: Sequence[float],
    policy_a_right_scores: Sequence[float],
    policy_b_left_scores: Sequence[float],
    policy_b_right_scores: Sequence[float],
    *,
    numeric_tolerance: float = 0.0,
) -> RateResult:
    """Return the fraction of pairs whose signed order differs by policy.

    Ties are represented by sign zero and therefore count as reversals when
    only one policy ties.  ``numeric_tolerance`` controls the tie boundary.
    """

    if isinstance(numeric_tolerance, bool) or not isinstance(
        numeric_tolerance, (int, float)
    ):
        raise TypeError("numeric_tolerance must be a real number")
    tolerance = float(numeric_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("numeric_tolerance must be finite and non-negative")

    _require_equal_lengths(
        policy_a_left_scores=policy_a_left_scores,
        policy_a_right_scores=policy_a_right_scores,
        policy_b_left_scores=policy_b_left_scores,
        policy_b_right_scores=policy_b_right_scores,
    )
    a_left = _validated_scores(policy_a_left_scores, name="policy_a_left_scores")
    a_right = _validated_scores(policy_a_right_scores, name="policy_a_right_scores")
    b_left = _validated_scores(policy_b_left_scores, name="policy_b_left_scores")
    b_right = _validated_scores(policy_b_right_scores, name="policy_b_right_scores")

    numerator = sum(
        _sign(left_a - right_a, tolerance) != _sign(left_b - right_b, tolerance)
        for left_a, right_a, left_b, right_b in zip(
            a_left, a_right, b_left, b_right, strict=True
        )
    )
    return _rate(numerator, len(a_left))


__all__ = [
    "DebtResult",
    "MetricStatus",
    "RateResult",
    "ScalarizationResult",
    "debt_grade_indices",
    "forced_order_debt",
    "policy_reversal_rate",
    "within_grade_compression_debt",
]

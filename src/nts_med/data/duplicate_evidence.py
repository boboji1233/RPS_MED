"""Versioned evidence policy for cross-split duplicate control.

Perceptual hashes and AI screens remain triage evidence.  Only exact identity,
an official duplicate mapping, or the preregistered frozen-feature threshold
can create a split-blocking duplicate state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

DUPLICATE_EVIDENCE_POLICY_VERSION: Final[str] = "2026-08-24-v1"
FROZEN_FEATURE_HIGH_RISK_THRESHOLD: Final[float] = 0.995


class DuplicateEvidenceState(StrEnum):
    """Fail-closed evidence states used before split sealing."""

    CONFIRMED_DUPLICATE = "CONFIRMED_DUPLICATE"
    HIGH_RISK_NEAR_DUPLICATE = "HIGH_RISK_NEAR_DUPLICATE"
    TRIAGE_CANDIDATE = "TRIAGE_CANDIDATE"
    NO_HIGH_CONFIDENCE_DUPLICATE_EVIDENCE = (
        "NO_HIGH_CONFIDENCE_DUPLICATE_EVIDENCE"
    )


@dataclass(frozen=True, slots=True)
class DuplicatePairEvidence:
    """Path-free evidence for one cross-split pair."""

    sha256_equal: bool = False
    normalized_pixel_sha256_equal: bool = False
    official_duplicate_mapping: bool = False
    perceptual_hash_candidate: bool = False
    frozen_feature_similarity: float | None = None

    def __post_init__(self) -> None:
        if self.frozen_feature_similarity is not None:
            similarity = self.frozen_feature_similarity
            if (
                isinstance(similarity, bool)
                or not isinstance(similarity, (int, float))
                or not math.isfinite(float(similarity))
                or not -1.0 <= float(similarity) <= 1.0
            ):
                raise ValueError("frozen_feature_similarity must be finite in [-1, 1]")


@dataclass(frozen=True, slots=True)
class DuplicateAuditCoverage:
    """Coverage required before a clean aggregate disposition can be issued."""

    exact_byte_hash_complete: bool
    normalized_pixel_hash_complete: bool
    frozen_feature_cross_split_complete: bool

    @property
    def complete(self) -> bool:
        return (
            self.exact_byte_hash_complete
            and self.normalized_pixel_hash_complete
            and self.frozen_feature_cross_split_complete
        )


def classify_duplicate_pair(
    evidence: DuplicatePairEvidence,
    *,
    high_risk_feature_threshold: float = FROZEN_FEATURE_HIGH_RISK_THRESHOLD,
) -> DuplicateEvidenceState:
    """Classify one pair without treating perceptual similarity as identity."""

    if (
        isinstance(high_risk_feature_threshold, bool)
        or not isinstance(high_risk_feature_threshold, (int, float))
        or not math.isfinite(float(high_risk_feature_threshold))
        or not -1.0 <= float(high_risk_feature_threshold) <= 1.0
    ):
        raise ValueError("high_risk_feature_threshold must be finite in [-1, 1]")
    if (
        evidence.sha256_equal
        or evidence.normalized_pixel_sha256_equal
        or evidence.official_duplicate_mapping
    ):
        return DuplicateEvidenceState.CONFIRMED_DUPLICATE
    if (
        evidence.frozen_feature_similarity is not None
        and evidence.frozen_feature_similarity >= float(high_risk_feature_threshold)
    ):
        return DuplicateEvidenceState.HIGH_RISK_NEAR_DUPLICATE
    if evidence.perceptual_hash_candidate or evidence.frozen_feature_similarity is not None:
        return DuplicateEvidenceState.TRIAGE_CANDIDATE
    return DuplicateEvidenceState.NO_HIGH_CONFIDENCE_DUPLICATE_EVIDENCE


def clean_audit_disposition(
    coverage: DuplicateAuditCoverage,
    *,
    confirmed_duplicate_count: int,
    high_risk_near_duplicate_count: int,
) -> DuplicateEvidenceState:
    """Return a clean state only after all independent audit channels complete."""

    for name, value in (
        ("confirmed_duplicate_count", confirmed_duplicate_count),
        ("high_risk_near_duplicate_count", high_risk_near_duplicate_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if not coverage.complete:
        raise ValueError("all exact and frozen-feature audit channels must be complete")
    if confirmed_duplicate_count:
        return DuplicateEvidenceState.CONFIRMED_DUPLICATE
    if high_risk_near_duplicate_count:
        return DuplicateEvidenceState.HIGH_RISK_NEAR_DUPLICATE
    return DuplicateEvidenceState.NO_HIGH_CONFIDENCE_DUPLICATE_EVIDENCE


__all__ = [
    "DUPLICATE_EVIDENCE_POLICY_VERSION",
    "FROZEN_FEATURE_HIGH_RISK_THRESHOLD",
    "DuplicateAuditCoverage",
    "DuplicateEvidenceState",
    "DuplicatePairEvidence",
    "classify_duplicate_pair",
    "clean_audit_disposition",
]

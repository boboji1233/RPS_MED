"""Interval relation algebra and scalarization information audits."""

from nts_med.relations.algebra import (
    Interval,
    Relation,
    RelationConfig,
    RelationEvidence,
    UnresolvedReason,
    infer_relation,
    infer_relation_with_evidence,
    unresolved_relation_evidence,
)
from nts_med.relations.pair_sampling import (
    DirectedPair,
    UnorderedPair,
    all_unordered_pairs,
    sample_directed_pairs,
)
from nts_med.relations.robustness import (
    PairStability,
    RelationStabilitySummary,
    comonotonic_burdens,
    independently_permuted_burdens,
    structural_relation_stability,
)
from nts_med.relations.scalarization import (
    DebtResult,
    MetricStatus,
    RateResult,
    ScalarizationResult,
    debt_grade_indices,
    forced_order_debt,
    policy_reversal_rate,
    within_grade_compression_debt,
)

__all__ = [
    "DebtResult",
    "DirectedPair",
    "Interval",
    "MetricStatus",
    "PairStability",
    "RateResult",
    "Relation",
    "RelationConfig",
    "RelationEvidence",
    "RelationStabilitySummary",
    "ScalarizationResult",
    "UnorderedPair",
    "UnresolvedReason",
    "all_unordered_pairs",
    "comonotonic_burdens",
    "debt_grade_indices",
    "forced_order_debt",
    "independently_permuted_burdens",
    "infer_relation",
    "infer_relation_with_evidence",
    "policy_reversal_rate",
    "sample_directed_pairs",
    "structural_relation_stability",
    "unresolved_relation_evidence",
    "within_grade_compression_debt",
]

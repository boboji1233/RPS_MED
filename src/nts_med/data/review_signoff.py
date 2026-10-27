"""Fail-closed validation for privacy-safe duplicate-review sign-off records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

_EXPECTED_PACK_COUNTS: dict[str, dict[str, int]] = {
    "maples_dr_messidor1": {
        "HASH_CANDIDATES": 726,
        "FROZEN_FEATURE_TOP_200": 200,
    },
    "ddr": {
        "PRIORITY_UNIQUE_PAIRS": 383,
        "WIDER_HASH_CANDIDATES": 10514,
    },
}

_REQUIRED_ATTESTATIONS = (
    "reviewed_only_local_privacy_safe_material",
    "no_raw_path_or_patient_identifier_recorded",
    "did_not_treat_ai_screen_as_human_review",
    "did_not_delete_or_move_images",
    "did_not_modify_split_assignments",
    "uncertain_pairs_left_unresolved",
)

_REQUIRED_BOUNDARIES: dict[str, bool] = {
    "split_change_authorized_by_this_form": False,
    "automatic_deletion_authorized_by_this_form": False,
    "gate0_authorized_by_this_form": False,
    "separate_split_sealing_decision_required": True,
}

_DECISION_FIELDS_BY_TEMPLATE: dict[int, tuple[str, ...]] = {
    1: ("visually_distinct", "likely_duplicate", "unresolved"),
    2: (
        "distinct",
        "likely_duplicate",
        "confirmed_duplicate",
        "unresolved",
        "excluded_from_review",
    ),
}

_FOLLOWUP_LIST_BY_TEMPLATE: dict[int, str] = {
    1: "opaque_pair_numbers_requiring_escalation",
    2: "opaque_pair_numbers_requiring_followup",
}


@dataclass(frozen=True, slots=True)
class SignoffValidation:
    """Privacy-safe validation result for one explicitly requested review scope."""

    scope: str
    complete: bool
    reviewed_pair_count: int
    decision_total: int
    error_codes: tuple[str, ...]


def _mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        return None
    return value


def _non_empty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _non_negative_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _valid_date(value: object) -> bool:
    if isinstance(value, date):
        return True
    if not isinstance(value, str):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _validate_identity(record: Mapping[str, object], errors: list[str]) -> None:
    reviewer = _mapping(record.get("reviewer"))
    if reviewer is None:
        errors.append("REVIEWER_MAPPING_REQUIRED")
        return
    for key in ("name_or_initials", "organization_or_role", "relevant_expertise"):
        if not _non_empty_text(reviewer.get(key)):
            errors.append(f"REVIEWER_{key.upper()}_REQUIRED")
    if not isinstance(reviewer.get("clinical_expert_claimed"), bool):
        errors.append("REVIEWER_CLINICAL_EXPERT_FLAG_REQUIRED")
    if not _valid_date(reviewer.get("review_date")):
        errors.append("REVIEW_DATE_REQUIRED")


def _validate_attestations(record: Mapping[str, object], errors: list[str]) -> None:
    attestations = _mapping(record.get("attestation"))
    if attestations is None:
        errors.append("ATTESTATION_MAPPING_REQUIRED")
        return
    for key in _REQUIRED_ATTESTATIONS:
        if attestations.get(key) is not True:
            errors.append(f"ATTESTATION_{key.upper()}_REQUIRED")


def _validate_signature(record: Mapping[str, object], errors: list[str]) -> None:
    signoff = _mapping(record.get("signoff"))
    if signoff is None:
        errors.append("SIGNOFF_MAPPING_REQUIRED")
        return
    for key in ("statement", "signed_by", "signature_method"):
        if not _non_empty_text(signoff.get(key)):
            errors.append(f"SIGNOFF_{key.upper()}_REQUIRED")
    if not _valid_date(signoff.get("signed_on")):
        errors.append("SIGNOFF_DATE_REQUIRED")


def _validate_boundaries(record: Mapping[str, object], errors: list[str]) -> None:
    boundaries = _mapping(record.get("post_review_boundary"))
    if boundaries is None:
        errors.append("POST_REVIEW_BOUNDARY_MAPPING_REQUIRED")
        return
    for key, expected in _REQUIRED_BOUNDARIES.items():
        if boundaries.get(key) is not expected:
            errors.append(f"POST_REVIEW_BOUNDARY_{key.upper()}_INVALID")


def _validate_packs(
    scope: Mapping[str, object],
    expected: Mapping[str, int],
    errors: list[str],
) -> int:
    packs = scope.get("packs")
    if not isinstance(packs, list):
        errors.append("PACKS_LIST_REQUIRED")
        return 0
    observed: dict[str, Mapping[str, object]] = {}
    for item in packs:
        pack = _mapping(item)
        if pack is None or not _non_empty_text(pack.get("name")):
            errors.append("PACK_MAPPING_OR_NAME_INVALID")
            continue
        name = str(pack["name"])
        if name in observed:
            errors.append("PACK_NAME_DUPLICATED")
            continue
        observed[name] = pack

    if set(observed) != set(expected):
        errors.append("PACK_SCOPE_MISMATCH")

    reviewed_total = 0
    for name, expected_count in expected.items():
        pack = observed.get(name)
        if pack is None:
            continue
        if pack.get("expected_pair_count") != expected_count:
            errors.append(f"PACK_{name}_EXPECTED_COUNT_MISMATCH")
        reviewed = pack.get("reviewed_pair_count")
        if not _non_negative_integer(reviewed):
            errors.append(f"PACK_{name}_REVIEWED_COUNT_INVALID")
            continue
        assert isinstance(reviewed, int)
        reviewed_total += reviewed
        if reviewed != expected_count:
            errors.append(f"PACK_{name}_REVIEW_INCOMPLETE")
    return reviewed_total


def _validate_decisions(
    scope: Mapping[str, object],
    reviewed_total: int,
    template_version: int,
    errors: list[str],
) -> int:
    decisions = _mapping(scope.get("decision_totals"))
    if decisions is None:
        errors.append("DECISION_TOTALS_MAPPING_REQUIRED")
        return 0
    expected_fields = _DECISION_FIELDS_BY_TEMPLATE[template_version]
    if set(decisions) != set(expected_fields):
        errors.append("DECISION_SCOPE_MISMATCH")
    total = 0
    followup_total = 0
    distinct_field = "visually_distinct" if template_version == 1 else "distinct"
    for key in expected_fields:
        value = decisions.get(key)
        if not _non_negative_integer(value):
            errors.append(f"DECISION_{key.upper()}_INVALID")
            continue
        assert isinstance(value, int)
        total += value
        if key != distinct_field:
            followup_total += value
    if total != reviewed_total:
        errors.append("DECISION_TOTAL_DOES_NOT_EQUAL_REVIEWED_COUNT")

    followup_field = _FOLLOWUP_LIST_BY_TEMPLATE[template_version]
    followup_ids = scope.get(followup_field)
    if not isinstance(followup_ids, list):
        errors.append(
            "ESCALATION_PAIR_NUMBERS_LIST_REQUIRED"
            if template_version == 1
            else "FOLLOWUP_PAIR_NUMBERS_LIST_REQUIRED"
        )
        return total
    followup_ids_are_valid = not any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in followup_ids
    )
    if not followup_ids_are_valid:
        errors.append(
            "ESCALATION_PAIR_NUMBER_INVALID"
            if template_version == 1
            else "FOLLOWUP_PAIR_NUMBER_INVALID"
        )
    elif len(set(followup_ids)) != len(followup_ids):
        errors.append(
            "ESCALATION_PAIR_NUMBER_DUPLICATED"
            if template_version == 1
            else "FOLLOWUP_PAIR_NUMBER_DUPLICATED"
        )
    if len(followup_ids) != followup_total:
        errors.append(
            "ESCALATION_COUNT_DOES_NOT_EQUAL_NON_DISTINCT_DECISIONS"
            if template_version == 1
            else "FOLLOWUP_COUNT_DOES_NOT_EQUAL_NON_DISTINCT_DECISIONS"
        )
    return total


def validate_duplicate_signoff_record(
    record: Mapping[str, object], scope_name: str
) -> SignoffValidation:
    """Validate one scope without exposing reviewer content or local source paths."""

    if scope_name not in _EXPECTED_PACK_COUNTS:
        raise ValueError("unsupported duplicate-review scope")
    errors: list[str] = []
    if record.get("schema_version") != 1:
        errors.append("SCHEMA_VERSION_INVALID")
    template_version = record.get("template_version")
    if template_version not in _DECISION_FIELDS_BY_TEMPLATE:
        errors.append("TEMPLATE_VERSION_INVALID")
        template_version = 1
    if record.get("status") not in {"SIGNED_PARTIAL", "SIGNED_COMPLETE"}:
        errors.append("SIGNED_RECORD_STATUS_REQUIRED")

    _validate_identity(record, errors)
    _validate_attestations(record, errors)
    _validate_signature(record, errors)
    _validate_boundaries(record, errors)

    scopes = _mapping(record.get("scopes"))
    scope = _mapping(scopes.get(scope_name)) if scopes is not None else None
    if scope is None:
        errors.append("REQUESTED_SCOPE_MAPPING_REQUIRED")
        return SignoffValidation(scope_name, False, 0, 0, tuple(errors))
    if scope.get("status") != "COMPLETE":
        errors.append("REQUESTED_SCOPE_STATUS_NOT_COMPLETE")

    reviewed_total = _validate_packs(scope, _EXPECTED_PACK_COUNTS[scope_name], errors)
    assert isinstance(template_version, int)
    decision_total = _validate_decisions(scope, reviewed_total, template_version, errors)
    return SignoffValidation(
        scope=scope_name,
        complete=not errors,
        reviewed_pair_count=reviewed_total,
        decision_total=decision_total,
        error_codes=tuple(errors),
    )


def load_duplicate_signoff_record(path: str | Path) -> dict[str, Any]:
    """Load one sign-off YAML mapping with path-safe failure messages."""

    try:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read sign-off record: {type(exc).__name__}") from exc
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise ValueError("sign-off record must be a YAML mapping")
    return loaded


__all__ = [
    "SignoffValidation",
    "load_duplicate_signoff_record",
    "validate_duplicate_signoff_record",
]

"""Atomic, privacy-safe consumption of one-run Gate 0 authorization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nts_med.reproducibility.freeze import sha256_file, validate_authorization


@dataclass(frozen=True, slots=True)
class AuthorizationClaim:
    """Aggregate receipt created before any held-out scientific access."""

    freeze_id: str
    passport_sha256: str
    authorization_sha256: str
    marker_sha256: str


def claim_one_run_authorization(
    authorization_path: str | Path,
    consumption_marker: str | Path,
    *,
    passport_sha256: str,
    freeze_id: str,
    authorization_scope: str = "FORMAL_GATE0_ONCE",
) -> AuthorizationClaim:
    """Validate and atomically claim one authorization exactly once.

    The marker is created with exclusive-create semantics. A pre-existing
    marker blocks access even if the previous execution later failed.
    """

    valid, reasons = validate_authorization(
        authorization_path,
        passport_sha256=passport_sha256,
        freeze_id=freeze_id,
        authorization_scope=authorization_scope,
    )
    if not valid:
        rendered = ",".join(reasons) if reasons else "UNKNOWN"
        raise PermissionError(f"formal authorization invalid: {rendered}")
    marker = Path(consumption_marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "FORMAL_AUTHORIZATION_CLAIMED_BEFORE_SCIENTIFIC_ACCESS",
        "authorization_scope": authorization_scope,
        "freeze_id": freeze_id,
        "passport_sha256": passport_sha256,
        "authorization_sha256": sha256_file(authorization_path),
        "gate0_execution_count_authorized": 1,
        "formal_execution_count_authorized": 1,
        "training": "FORBIDDEN",
        "raw_paths_recorded": False,
        "source_identifiers_recorded": False,
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with marker.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
    except FileExistsError as error:
        raise PermissionError("formal authorization was already claimed") from error
    return AuthorizationClaim(
        freeze_id=freeze_id,
        passport_sha256=passport_sha256,
        authorization_sha256=str(payload["authorization_sha256"]),
        marker_sha256=sha256_file(marker),
    )


__all__ = ["AuthorizationClaim", "claim_one_run_authorization"]

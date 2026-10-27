"""Validate a privacy-safe human duplicate-review sign-off without changing data."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from nts_med.cli._common import print_status
from nts_med.data.review_signoff import (
    load_duplicate_signoff_record,
    validate_duplicate_signoff_record,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed validation of one human duplicate-review sign-off scope."
    )
    parser.add_argument("--record", required=True, help="Privacy-safe versioned YAML record")
    parser.add_argument(
        "--scope",
        required=True,
        choices=("maples_dr_messidor1", "ddr"),
        help="Review scope that must be complete",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        record = load_duplicate_signoff_record(args.record)
        result = validate_duplicate_signoff_record(record, args.scope)
    except ValueError as exc:
        print_status(
            {
                "status": "SIGNOFF_SCOPE_BLOCKED",
                "scope": args.scope,
                "error_codes": [str(exc)],
                "gate0_authorized": False,
                "split_change_authorized": False,
            }
        )
        return 2

    print_status(
        {
            "status": (
                "SIGNOFF_SCOPE_VALIDATED" if result.complete else "SIGNOFF_SCOPE_BLOCKED"
            ),
            "scope": result.scope,
            "complete": result.complete,
            "reviewed_pair_count": result.reviewed_pair_count,
            "decision_total": result.decision_total,
            "error_codes": list(result.error_codes),
            "gate0_authorized": False,
            "split_change_authorized": False,
        }
    )
    return 0 if result.complete else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""CLI for pure-synthetic validation of Gate 0 preregistration v0.6."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from nts_med.synthetic.gate0_v06 import (
    SYNTHETIC_GATE0_V06_STATUS,
    run_synthetic_gate0_v06,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Gate 0 v0.6 rules using constructed inputs only."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/audits/synthetic_gate0_v06"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--permutation-repetitions", type=int, default=5000)
    parser.add_argument("--master-seed", type=int, default=20260823)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_synthetic_gate0_v06(
            args.output,
            bootstrap_replicates=args.bootstrap_replicates,
            permutation_repetitions=args.permutation_repetitions,
            master_seed=args.master_seed,
        )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, sort_keys=True))
    return int(
        summary["status"] != SYNTHETIC_GATE0_V06_STATUS
        or summary["all_checks_passed"] is not True
    )


if __name__ == "__main__":
    raise SystemExit(main())

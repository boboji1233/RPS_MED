"""Command-line entry point for deterministic synthetic relation validation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from nts_med.synthetic import (
    DEFAULT_PAIRS_PER_RELATION,
    DEFAULT_SEED,
    SYNTHETIC_STATUS,
    run_synthetic_smoke,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser so help and parsing can be tested without subprocesses."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic synthetic intervals and verify all five "
            "NTS-Med relations. This does not run a scientific experiment."
        )
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/feasibility/synthetic_smoke"),
        help="artifact directory",
    )
    parser.add_argument(
        "--axes",
        "--num-axes",
        dest="num_axes",
        type=int,
        default=4,
        help="number of synthetic pathology axes (minimum: 2)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.05,
        help="non-negative coordinate perturbation amplitude",
    )
    parser.add_argument(
        "--pairs-per-relation",
        type=int,
        default=DEFAULT_PAIRS_PER_RELATION,
        help="directed pairs per relation (minimum: 50)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smoke test, print its aggregate JSON, and return a process status."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_synthetic_smoke(
            args.output,
            seed=args.seed,
            num_axes=args.num_axes,
            noise=args.noise,
            pairs_per_relation=args.pairs_per_relation,
        )
    except ValueError as error:
        parser.error(str(error))

    print(json.dumps(summary, sort_keys=True))
    valid_status = summary["status"] == SYNTHETIC_STATUS
    all_matched = summary["all_expected_relations_matched"] is True
    return 0 if valid_status and all_matched else 1


if __name__ == "__main__":
    raise SystemExit(main())

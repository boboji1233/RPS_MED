"""CLI for the synthetic-only full Gate 0 dry-run."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from nts_med.relations.algebra import RelationConfig
from nts_med.synthetic.gate0_dry_run import (
    SYNTHETIC_GATE0_DRY_RUN_STATUS,
    run_synthetic_gate0_dry_run,
)


def _mapping(snapshot: Mapping[str, object], key: str) -> dict[str, float]:
    value = snapshot.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return {str(axis): float(threshold) for axis, threshold in value.items()}


def _load_relation_config(path: Path) -> RelationConfig:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("relation config must be a mapping")
    return RelationConfig(
        delta=_mapping(loaded, "delta"),
        epsilon=_mapping(loaded, "epsilon"),
        max_width=_mapping(loaded, "max_width"),
        numeric_tolerance=float(loaded.get("numeric_tolerance", 1.0e-12)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the synthetic Gate 0 pipeline; never reads real data."
    )
    parser.add_argument(
        "--relation-config", type=Path,
        default=Path("configs/relation/dr_four_axis.yaml"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/audits/synthetic_gate0_dry_run"),
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_synthetic_gate0_dry_run(
            args.output,
            _load_relation_config(args.relation_config),
            seed=args.seed,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, sort_keys=True))
    return int(
        summary["status"] != SYNTHETIC_GATE0_DRY_RUN_STATUS
        or summary["all_checks_passed"] is not True
    )


if __name__ == "__main__":
    raise SystemExit(main())

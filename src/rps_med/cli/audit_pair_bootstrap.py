"""Write a development-only complete-graph pair-bootstrap audit."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rps_med.cli.formal_r1 import _write_once
from rps_med.pair_bootstrap_audit import audit_development_pair_bootstrap


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights-directory", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260832)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError("pair bootstrap audit output already exists")
    result = audit_development_pair_bootstrap(
        args.cache,
        args.config,
        args.weights_directory,
        replicates=args.replicates,
        seed=args.seed,
    )
    _write_once(args.output, {"schema_version": 1, **result})
    print(json.dumps({"status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the frozen RPS-Med Stage 3A development-only matrix once."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rps_med.stage3a_execution import run_stage3a_development


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--pretrained-weights", type=Path, required=True)
    parser.add_argument("--common-config", type=Path, required=True)
    parser.add_argument("--runs-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--include-p1", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_stage3a_development(
        args.repository_root,
        args.cache,
        args.pretrained_weights,
        args.common_config,
        args.runs_config,
        args.output_root,
        include_p1=args.include_p1,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

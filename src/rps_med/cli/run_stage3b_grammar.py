"""Command-line entry point for the frozen RPS-Med Stage 3B grammar study."""

from __future__ import annotations

import argparse
import json

from rps_med.stage3b_execution import run_stage3b_grammar


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--stage3a-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_stage3b_grammar(
        args.repository_root,
        args.config,
        args.cache,
        args.stage3a_root,
        args.output,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

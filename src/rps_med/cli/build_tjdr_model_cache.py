"""Build a write-once, leakage-controlled TJDR development model cache."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rps_med.model_data import build_tjdr_development_cache, cache_summary_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-archive", type=Path, action="append", required=True)
    parser.add_argument("--test-archive", type=Path, action="append", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=20260826)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.summary.exists():
        raise FileExistsError("TJDR model cache summary already exists")
    summary = build_tjdr_development_cache(
        args.train_archive,
        args.test_archive,
        args.cache,
        image_size=args.image_size,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(cache_summary_json(summary), encoding="utf-8")
    print(json.dumps({"status": summary["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

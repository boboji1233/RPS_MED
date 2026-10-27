"""Write-once aggregate TJDR train/test image leakage audit."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rps_med.cli.formal_r1 import _write_once
from rps_med.tjdr_leakage import audit_tjdr_image_leakage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-archive", type=Path, action="append", required=True)
    parser.add_argument("--test-archive", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError("TJDR leakage audit output already exists")
    payload = {
        "schema_version": 2,
        **audit_tjdr_image_leakage(args.train_archive, args.test_archive),
        "analysis_role": "MODEL_LEAKAGE_SCREENING_ONLY",
        "model_training": "NOT_RUN",
    }
    _write_once(args.output, payload)
    print(json.dumps({"status": payload["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

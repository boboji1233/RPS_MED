"""Write a path-free aggregate TJDR/DDR model-overlap audit."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rps_med.cli.formal_r1 import _write_once
from rps_med.cross_source_leakage import audit_tjdr_ddr_model_overlap


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tjdr-train-archive", type=Path, action="append", required=True)
    parser.add_argument("--tjdr-test-archive", type=Path, action="append", required=True)
    parser.add_argument("--ddr-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError("cross-source overlap audit output already exists")
    payload = {
        "schema_version": 1,
        **audit_tjdr_ddr_model_overlap(
            args.tjdr_train_archive, args.tjdr_test_archive, args.ddr_root
        ),
        "analysis_role": "EXTERNAL_TRANSFER_LEAKAGE_SCREENING_ONLY",
        "model_training": "NOT_RUN",
    }
    _write_once(args.output, payload)
    print(json.dumps({"status": payload["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

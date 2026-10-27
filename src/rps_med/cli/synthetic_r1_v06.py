"""Run deterministic synthetic-only checks of the RPS-Med R1 v0.6 diagnostics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rps_med.cli.formal_r1_v06 import _load_v06_config
from rps_med.source_io_v06 import (
    DDRDiagnosticState,
    DDRExclusionReason,
    DDRSourceMeasurementError,
    DDRSubstage,
    _run_substage,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _fail_synthetically() -> None:
    raise ValueError("synthetic detail forbidden from output")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repository_root.resolve()
    _load_v06_config(root, args.config)

    stage_matrix: list[dict[str, object]] = []
    for substage in DDRSubstage:
        state = DDRDiagnosticState()
        try:
            _run_substage(state, substage, _fail_synthetically)
        except DDRSourceMeasurementError as error:
            stage_matrix.append(
                {
                    "failed_substage": error.substage.value,
                    "error_family": error.error_family,
                    "exception_message_recorded": error.diagnostics["exception_message_recorded"],
                    "source_identifiers_recorded": error.diagnostics["source_identifiers_recorded"],
                }
            )

    state = DDRDiagnosticState()
    for reason in DDRExclusionReason:
        state.exclude(reason)
    payload = {
        "schema_version": 1,
        "status": "RPS_MED_R1_V0_6_SYNTHETIC_DIAGNOSTIC_COMPLETE",
        "substage_fault_matrix": stage_matrix,
        "exclusion_reason_matrix": state.safe_diagnostics(),
        "formal_r1": "NOT_RUN_SYNTHETIC_ONLY",
        "scientific_data_opened": False,
        "raw_input_bytes_opened": False,
        "gate0": "NOT_RERUN",
        "training": "FORBIDDEN_NOT_RUN",
        "local_paths_recorded": False,
        "source_identifiers_recorded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

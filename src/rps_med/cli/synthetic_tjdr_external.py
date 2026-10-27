"""Deterministic synthetic validation for the TJDR external gate contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from nts_med.gate0.execution import MeasurementRecord, SourceMeasurements
from rps_med.cli.formal_r1 import _write_once, _yaml_mapping
from rps_med.external import execute_tjdr_external_gate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _source(name: str, role: str) -> SourceMeasurements:
    records = tuple(
        MeasurementRecord(
            instance_id=f"synthetic-{name}-{index}",
            primary={
                "ma": float(index),
                "hem": float(30 - index),
                "ex": float((index % 5) * 4),
                "cws": float((index % 3) * 6),
            },
            variants={},
            grade=None,
            grading_split_conflict=False,
            strict_measurement_quality=True,
        )
        for index in range(20)
    )
    return SourceMeasurements(name, role, len(records), len(records), records, 0)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError("synthetic TJDR external output already exists")
    config = _yaml_mapping(args.config)
    development = _source("development", "DEVELOPMENT_CALIBRATION")
    test = _source("test", "INDEPENDENT_EXTERNAL_REPLICATION")
    first = execute_tjdr_external_gate(development, test, config)
    second = execute_tjdr_external_gate(development, test, config)
    exclusion_fail_closed = False
    invalid = SourceMeasurements(
        test.name,
        test.role,
        test.expected_input_count,
        test.observed_input_count,
        test.records,
        1,
    )
    try:
        execute_tjdr_external_gate(development, invalid, config)
    except ValueError:
        exclusion_fail_closed = True
    checks = {
        "repeat_result_equal": first == second,
        "five_criteria_present": len(first["criteria"]) == 5,
        "four_stochastic_endpoints": (
            first["uncertainty"]["stochastic_core_endpoint_count"] == 4
        ),
        "measurement_exclusion_fail_closed": exclusion_fail_closed,
        "pair_rows_not_persisted": first["pair_rows_persisted"] is False,
        "bootstrap_rows_not_persisted": (
            first["uncertainty"]["bootstrap_rows_persisted"] is False
        ),
    }
    payload = {
        "schema_version": 1,
        "status": (
            "RPS_MED_TJDR_EXTERNAL_SYNTHETIC_VALIDATION_PASS"
            if all(checks.values())
            else "RPS_MED_TJDR_EXTERNAL_SYNTHETIC_VALIDATION_FAIL"
        ),
        "checks": checks,
        "synthetic_result": first,
        "real_scientific_data_opened": False,
        "external_replication_gate": "NOT_RUN",
        "gate0": "NOT_RERUN",
        "training": "FORBIDDEN_NOT_RUN",
        "local_paths_recorded": False,
        "source_identifiers_recorded": False,
    }
    _write_once(args.output, payload)
    print(json.dumps({"status": payload["status"]}))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

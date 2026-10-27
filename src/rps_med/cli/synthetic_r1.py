"""Run deterministic positive and negative synthetic controls for RPS-Med R1."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from nts_med.gate0.execution import MeasurementRecord, SourceMeasurements
from rps_med.formal import execute_formal_r1


def _record(identifier: str, first: float, second: float) -> MeasurementRecord:
    return MeasurementRecord(
        instance_id=identifier,
        primary={"ma": first, "hem": second, "ex": first, "cws": second},
        variants={},
        grade=None,
        grading_split_conflict=False,
        strict_measurement_quality=True,
    )


def _source(name: str, *, negative: bool) -> SourceMeasurements:
    if negative:
        points = ((1.0, 1.0),) * 60
    else:
        base_points = (
            (0.0, 12.0),
            (12.0, 0.0),
            (0.0, 10.0),
            (10.0, 0.0),
            (1.0, 13.0),
            (13.0, 1.0),
            (2.0, 14.0),
            (14.0, 2.0),
            (3.0, 15.0),
            (15.0, 3.0),
        )
        points = base_points * 6
    records = tuple(
        _record(f"synthetic_{name}_{index}", first, second)
        for index, (first, second) in enumerate(points)
    )
    return SourceMeasurements(
        name=name,
        role="SYNTHETIC_CONTROL",
        expected_input_count=len(records),
        observed_input_count=len(records),
        records=records,
        measurement_exclusion_count=0,
    )


def _summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision": result["decision"],
        "criteria": result["criteria"],
        "uncertainty": result["uncertainty"],
        "cross_source": result["cross_source"],
        "pair_rows_persisted": result["pair_rows_persisted"],
        "training": result["training"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    loaded = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("synthetic R1 config must be a mapping")
    positive = execute_formal_r1(
        _source("maples_dr_messidor1", negative=False),
        _source("ddr", negative=False),
        loaded,
    )
    negative = execute_formal_r1(
        _source("maples_dr_messidor1", negative=True),
        _source("ddr", negative=True),
        loaded,
    )
    payload = {
        "schema_version": 1,
        "status": "RPS_MED_R1_V0_5_SYNTHETIC_COMPLETE",
        "positive_control": _summary(positive),
        "negative_control": _summary(negative),
        "formal_r1": "NOT_RUN_SYNTHETIC_ONLY",
        "scientific_data_opened": False,
        "gate0": "NOT_RERUN",
        "training": "FORBIDDEN_NOT_RUN",
        "pair_rows_persisted": False,
        "bootstrap_rows_persisted": False,
        "local_paths_recorded": False,
        "source_identifiers_recorded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    print(json.dumps({"status": payload["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

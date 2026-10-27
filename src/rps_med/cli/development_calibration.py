"""Preflight or run the aggregate-only RPS-Med R1.2 development analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from rps_med.development import analyse_source, base_relation_config, cross_source_summary
from rps_med.source_io import (
    DevelopmentBoundary,
    inspect_development_boundary,
    load_ddr_development_measurements,
    load_maples_development_measurements,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-lock", type=Path)
    parser.add_argument("--maples-archive", type=Path, required=True)
    parser.add_argument("--maples-additional", type=Path, required=True)
    parser.add_argument("--ddr-root", type=Path, required=True)
    parser.add_argument("--ddr-private-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_mapping(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("YAML document must be a mapping")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_hashes(args: argparse.Namespace) -> dict[str, str]:
    return {
        "maples_main_archive": _sha256(args.maples_archive),
        "maples_additional_archive": _sha256(args.maples_additional),
        "ddr_private_manifest": _sha256(args.ddr_private_manifest),
    }


def _boundary_payload(boundary: DevelopmentBoundary) -> dict[str, Any]:
    return asdict(boundary)


def _expected_development_count(config: Mapping[str, Any]) -> int:
    boundary = config.get("development_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("development_boundary must be a mapping")
    return int(boundary["maples_development_count"])


def _verify_lock(
    lock: Mapping[str, Any],
    hashes: Mapping[str, str],
    boundary: DevelopmentBoundary,
) -> None:
    expected_hashes = lock.get("input_identity")
    expected_boundary = lock.get("development_boundary")
    if not isinstance(expected_hashes, Mapping) or not isinstance(
        expected_boundary, Mapping
    ):
        raise ValueError("input lock schema is invalid")
    if dict(expected_hashes) != dict(hashes):
        raise ValueError("input identity differs from the frozen lock")
    if dict(expected_boundary) != _boundary_payload(boundary):
        raise ValueError("development boundary differs from the frozen lock")
    if lock.get("held_out_or_test_scientific_access") != "FORBIDDEN":
        raise ValueError("input lock does not forbid held-out/test access")


def _grid(config: Mapping[str, Any]) -> tuple[tuple[float, ...], int]:
    scale = config.get("primary_scale_family")
    if not isinstance(scale, Mapping) or scale.get("policy") != "MARGIN_ONLY":
        raise ValueError("primary scale family must be MARGIN_ONLY")
    fine = tuple(float(value) for value in scale["fine_scales"])
    coarse_stride = int(scale["coarse_stride"])
    if len(fine) != 33 or coarse_stride != 2:
        raise ValueError("R1.2 requires the frozen 33/17 nested grids")
    return fine, coarse_stride


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError("output already exists; overwrite is forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _preflight(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    hashes = _input_hashes(args)
    boundary = inspect_development_boundary(
        args.maples_archive,
        args.ddr_root,
        maples_development_count=_expected_development_count(config),
    )
    payload = {
        "schema_version": 1,
        "status": "RPS_MED_R1_2_PREFLIGHT_COMPLETE",
        "input_identity": hashes,
        "development_boundary": _boundary_payload(boundary),
        "scientific_images_or_masks_opened": False,
        "held_out_or_test_scientific_access": False,
        "relations_computed": False,
        "pair_rows_persisted": False,
        "formal_r1": "NOT_RUN",
        "training": "FORBIDDEN",
        "local_paths_recorded": False,
        "source_identifiers_recorded": False,
    }
    _write_once(args.output, payload)
    print(json.dumps({"status": payload["status"]}))
    return 0


def _run(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    if args.input_lock is None:
        raise ValueError("--input-lock is required in run mode")
    stage = "INPUT_LOCK_VERIFICATION"
    try:
        hashes = _input_hashes(args)
        boundary = inspect_development_boundary(
            args.maples_archive,
            args.ddr_root,
            maples_development_count=_expected_development_count(config),
        )
        _verify_lock(_load_mapping(args.input_lock), hashes, boundary)

        stage = "DEVELOPMENT_SOURCE_MEASUREMENT"
        maples = load_maples_development_measurements(
            args.maples_archive,
            args.maples_additional,
            development_count=boundary.maples_development_count,
        )
        ddr = load_ddr_development_measurements(args.ddr_root)

        stage = "RELATION_PERSISTENCE_AGGREGATION"
        fine_scales, coarse_stride = _grid(config)
        relation_config = base_relation_config(config)
        maples_result = analyse_source(
            maples,
            base_config=relation_config,
            fine_scales=fine_scales,
            coarse_stride=coarse_stride,
            minimum_scale=fine_scales[0],
            maximum_scale=fine_scales[-1],
        )
        ddr_result = analyse_source(
            ddr,
            base_config=relation_config,
            fine_scales=fine_scales,
            coarse_stride=coarse_stride,
            minimum_scale=fine_scales[0],
            maximum_scale=fine_scales[-1],
        )

        stage = "CROSS_SOURCE_SUMMARY"
        cross_source = cross_source_summary(maples_result, ddr_result)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "status": "RPS_MED_R1_2_DEVELOPMENT_COMPLETE",
            "analysis_role": "DEVELOPMENT_CALIBRATION_ONLY",
            "input_lock_id": _load_mapping(args.input_lock).get("lock_id"),
            "sources": {
                "maples_dr_messidor1": maples_result,
                "ddr": ddr_result,
            },
            "cross_source_stability": cross_source,
            "held_out_or_test_scientific_access": False,
            "pair_rows_persisted": False,
            "formal_r1": "NOT_RUN",
            "gate0": "NOT_RERUN",
            "training": "FORBIDDEN_NOT_RUN",
            "local_paths_recorded": False,
            "source_identifiers_recorded": False,
        }
        _write_once(args.output, payload)
        print(json.dumps({"status": payload["status"]}))
        return 0
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        payload = {
            "schema_version": 1,
            "status": "RPS_MED_R1_2_DEVELOPMENT_ABORTED_TECHNICAL",
            "failed_stage": stage,
            "error_family": type(error).__name__.upper(),
            "reason_code": f"TECHNICAL_ERROR:{stage}",
            "exception_message_recorded": False,
            "held_out_or_test_scientific_access": False,
            "pair_rows_persisted": False,
            "formal_r1": "NOT_RUN",
            "training": "FORBIDDEN_NOT_RUN",
            "local_paths_recorded": False,
            "source_identifiers_recorded": False,
        }
        _write_once(args.output, payload)
        print(json.dumps({"status": payload["status"], "failed_stage": stage}))
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError("output already exists; overwrite is forbidden")
    config = _load_mapping(args.config)
    if args.mode == "preflight":
        return _preflight(args, config)
    return _run(args, config)


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit or analyse the official TJDR train split without opening test science."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

import yaml

from rps_med.cli.formal_r1 import _write_once
from rps_med.development import analyse_source, base_relation_config
from rps_med.tjdr_io import (
    audit_tjdr_train,
    inspect_tjdr_structure,
    load_tjdr_train_measurements,
)


class _FOVOptions(TypedDict):
    background_threshold: float
    minimum_area_ratio: float
    maximum_area_ratio: float
    substantial_component_ratio: float
    dominant_component_min_share: float


class _AnnotationOptions(TypedDict):
    allow_all_background: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "analyse"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-archive", type=Path, action="append", required=True)
    parser.add_argument("--audit-lock", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _load(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(value, name="config")


def _fov(config: Mapping[str, Any]) -> _FOVOptions:
    raw = _mapping(config.get("fov_candidate"), name="fov_candidate")
    return {
        "background_threshold": float(raw["background_threshold"]),
        "minimum_area_ratio": float(raw["minimum_area_ratio"]),
        "maximum_area_ratio": float(raw["maximum_area_ratio"]),
        "substantial_component_ratio": float(raw["substantial_component_ratio"]),
        "dominant_component_min_share": float(raw["dominant_component_min_share"]),
    }


def _annotation(config: Mapping[str, Any]) -> _AnnotationOptions:
    raw = _mapping(config.get("annotation"), name="annotation")
    return {
        "allow_all_background": bool(raw["allow_all_background_complete_annotation"])
    }


def _expected(config: Mapping[str, Any]) -> tuple[int, tuple[str, ...]]:
    source = _mapping(config.get("source"), name="source")
    hashes = tuple(str(value) for value in source["train_archive_sha256"])
    return int(source["expected_development_count"]), hashes


def _grid(config: Mapping[str, Any]) -> tuple[tuple[float, ...], int]:
    raw = _mapping(config.get("scale"), name="scale")
    minimum = float(raw["log2_minimum"])
    step = float(raw["sensitivity_step"])
    points = int(raw["sensitivity_points"])
    scales = tuple(2.0 ** (minimum + index * step) for index in range(points))
    stride = round(float(raw["primary_step"]) / step)
    if points != 33 or stride != 2 or not math.isclose(math.log2(scales[-1]), 2.0):
        raise ValueError("TJDR development scale grid is invalid")
    return scales, stride


def _assert_identity(
    config: Mapping[str, Any], archive_paths: Sequence[Path]
) -> tuple[int, dict[str, Any]]:
    expected_count, expected_hashes = _expected(config)
    boundary = inspect_tjdr_structure(archive_paths, split="train")
    if boundary.archive_sha256 != expected_hashes:
        raise ValueError("TJDR train archive identity mismatch")
    if boundary.image_count != expected_count or boundary.annotation_count != expected_count:
        raise ValueError("TJDR train structural count mismatch")
    return expected_count, {
        "split": boundary.split,
        "archive_count": boundary.archive_count,
        "archive_sha256": list(boundary.archive_sha256),
        "combined_member_count": boundary.combined_member_count,
        "image_count": boundary.image_count,
        "annotation_count": boundary.annotation_count,
        "opaque_pair_digest": boundary.opaque_pair_digest,
    }


def _audit(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    expected_count, boundary = _assert_identity(config, args.train_archive)
    audit = audit_tjdr_train(
        args.train_archive,
        expected_count=expected_count,
        **_fov(config),
        **_annotation(config),
    )
    payload = {
        "schema_version": 1,
        "status": audit["status"],
        "analysis_role": "TJDR_OFFICIAL_TRAIN_FORMAT_AND_FOV_DEVELOPMENT_ONLY",
        "boundary": boundary,
        "audit": audit,
        "test_scientific_access": False,
        "relations_computed": False,
        "external_replication_gate": "NOT_RUN",
        "training": "FORBIDDEN_NOT_RUN",
        "local_paths_recorded": False,
        "source_identifiers_recorded": False,
    }
    _write_once(args.output, payload)
    print(json.dumps({"status": payload["status"]}))
    return 0 if audit["status"] == "TJDR_TRAIN_AUDIT_PASS" else 2


def _analyse(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    if args.audit_lock is None:
        raise ValueError("--audit-lock is required in analyse mode")
    audit_lock = json.loads(args.audit_lock.read_text(encoding="utf-8"))
    if audit_lock.get("status") != "TJDR_TRAIN_AUDIT_PASS":
        raise ValueError("TJDR train audit did not pass")
    expected_count, boundary = _assert_identity(config, args.train_archive)
    if audit_lock.get("boundary") != boundary:
        raise ValueError("TJDR train audit boundary mismatch")
    source = load_tjdr_train_measurements(
        args.train_archive,
        expected_count=expected_count,
        **_fov(config),
        **_annotation(config),
    )
    if source.measurement_exclusion_count or len(source.records) != expected_count:
        raise ValueError("TJDR development measurement is incomplete")
    scales, stride = _grid(config)
    result = analyse_source(
        source,
        base_config=base_relation_config(config),
        fine_scales=scales,
        coarse_stride=stride,
        minimum_scale=scales[0],
        maximum_scale=scales[-1],
    )
    payload = {
        "schema_version": 1,
        "status": "TJDR_TRAIN_RELATION_PERSISTENCE_DEVELOPMENT_COMPLETE",
        "analysis_role": "DEVELOPMENT_CALIBRATION_ONLY",
        "boundary": boundary,
        "source": result,
        "test_scientific_access": False,
        "pair_rows_persisted": False,
        "external_replication_gate": "NOT_RUN",
        "training": "FORBIDDEN_NOT_RUN",
        "local_paths_recorded": False,
        "source_identifiers_recorded": False,
    }
    _write_once(args.output, payload)
    print(json.dumps({"status": payload["status"]}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError("TJDR development output already exists")
    config = _load(args.config)
    return _audit(args, config) if args.mode == "audit" else _analyse(args, config)


if __name__ == "__main__":
    raise SystemExit(main())

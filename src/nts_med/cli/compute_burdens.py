"""Compute four-axis DR burdens from an explicitly supplied NPZ mask bundle."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from nts_med.cli._common import (
    load_yaml_mapping,
    print_status,
    software_versions,
    unavailable_payload,
    write_json,
)
from nts_med.concepts.burden import DR_LESION_AXES, compute_dr_burden, validate_binary_mask


def build_parser() -> argparse.ArgumentParser:
    """Return the parser for the explicit NPZ adapter bridge."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute one four-axis burden record from NPZ arrays. This command does not "
            "discover or download a real dataset."
        )
    )
    parser.add_argument("--config", type=Path, default=None, help="optional YAML settings")
    parser.add_argument("--input-npz", type=Path, default=None, help="NPZ containing fov and masks")
    parser.add_argument("--output", type=Path, required=True, help="output .json or .csv")
    parser.add_argument(
        "--instance-id",
        "--sample-id",
        dest="instance_id",
        default="sample",
        help="non-patient output identifier",
    )
    return parser


def _axis_mapping(
    config: Mapping[str, Any],
    key: str,
    default: object,
) -> dict[str, Any]:
    raw = config.get(key, {})
    if not isinstance(raw, Mapping):
        raise ValueError(f"{key} must be a mapping")
    extra = set(raw) - set(DR_LESION_AXES)
    if extra:
        raise ValueError(f"{key} contains unexpected axes: {sorted(extra)}")
    return {axis: raw.get(axis, default) for axis in DR_LESION_AXES}


def _write_burden_output(
    output: Path,
    instance_id: str,
    burdens: Mapping[str, float],
    metadata: Mapping[str, object],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        write_json(output, {**metadata, "instance_id": instance_id, "burdens": dict(burdens)})
        return
    if output.suffix.lower() != ".csv":
        raise ValueError("burden output must end in .json or .csv during M0-M1")
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["instance_id", *DR_LESION_AXES],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow({"instance_id": instance_id, **burdens})
    write_json(Path(f"{output}.metadata.json"), metadata)


def main(argv: Sequence[str] | None = None) -> int:
    """Load explicitly keyed masks, validate them, and persist their burdens."""

    args = build_parser().parse_args(argv)
    try:
        config: dict[str, Any] = load_yaml_mapping(args.config) if args.config else {}
        configured_input = config.get("input_npz")
        input_path = args.input_npz or (Path(configured_input) if configured_input else None)
        if input_path is None or not input_path.is_file():
            print_status(unavailable_payload(input_path or "UNSPECIFIED", kind="mask_bundle"))
            return 0

        lesion_keys = _axis_mapping(config, "lesion_keys", "")
        lesion_keys = {
            axis: (value if isinstance(value, str) and value else axis)
            for axis, value in lesion_keys.items()
        }
        minimums = _axis_mapping(config, "min_component_pixels", 1)
        scales = _axis_mapping(config, "scale", 1.0)
        classes = config.get("positive_classes", {})
        if not isinstance(classes, Mapping):
            raise ValueError("positive_classes must be a mapping")
        extra_classes = set(classes) - set(DR_LESION_AXES)
        if extra_classes:
            raise ValueError(
                f"positive_classes contains unexpected axes: {sorted(extra_classes)}"
            )
        fov_key = config.get("fov_key", "fov")
        if not isinstance(fov_key, str) or not fov_key:
            raise ValueError("fov_key must be a non-empty string")

        with np.load(input_path, allow_pickle=False) as arrays:
            if fov_key not in arrays:
                raise ValueError(f"NPZ is missing FOV key {fov_key!r}")
            fov = arrays[fov_key]
            masks: dict[str, np.ndarray[Any, Any]] = {}
            for axis in DR_LESION_AXES:
                key = lesion_keys[axis]
                if key not in arrays:
                    raise ValueError(f"NPZ is missing lesion key {key!r} for {axis}")
                raw_mask = arrays[key]
                class_value = classes.get(axis)
                masks[axis] = validate_binary_mask(
                    raw_mask,
                    positive_class=class_value,
                    name=f"{axis}_mask",
                )

        burdens = compute_dr_burden(
            masks,
            fov,
            {axis: int(minimums[axis]) for axis in DR_LESION_AXES},
            scale={axis: float(scales[axis]) for axis in DR_LESION_AXES},
        )
        metadata: dict[str, object] = {
            "status": "BURDENS_COMPUTED",
            "input": str(input_path),
            "config_snapshot": {
                "fov_key": fov_key,
                "lesion_keys": lesion_keys,
                "min_component_pixels": minimums,
                "scale": scales,
                "positive_classes": dict(classes),
            },
            "software_versions": software_versions(),
        }
        _write_burden_output(args.output, args.instance_id, burdens, metadata)
    except (OSError, TypeError, ValueError) as error:
        print_status(
            {
                "status": "BURDEN_COMPUTATION_FAILED",
                "error": f"{type(error).__name__}: {error}",
                "software_versions": software_versions(),
            }
        )
        return 2
    print_status(metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

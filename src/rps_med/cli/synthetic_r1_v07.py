"""Run deterministic synthetic-only checks of the RPS-Med R1 v0.7 FOV amendment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from nts_med.concepts.fov import FOVStatus, extract_ddr_retinal_fov
from rps_med.cli.formal_r1 import _write_once
from rps_med.config_v07 import load_v07_config
from rps_med.fov_v07 import extract_ddr_retinal_fov_v07


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _image(components: tuple[tuple[slice, slice], ...]) -> NDArray[np.uint8]:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    for rows, columns in components:
        image[rows, columns] = 80
    return image


def _case(image: NDArray[np.uint8], *, dominance: float) -> dict[str, object]:
    legacy = extract_ddr_retinal_fov(image)
    amended = extract_ddr_retinal_fov_v07(
        image, dominant_component_min_share=dominance
    )
    masks_equal = (
        legacy.mask is not None
        and amended.mask is not None
        and np.array_equal(legacy.mask, amended.mask)
    )
    return {
        "legacy_status": legacy.status.value,
        "amended_status": amended.status.value,
        "amended_method": amended.method,
        "masks_equal": masks_equal,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repository_root.resolve()
    config = load_v07_config(root, args.config)
    dominance = float(config["measurement_amendment"]["dominant_component_min_share"])
    cases = {
        "legacy_single": _case(
            _image(((slice(10, 90), slice(10, 90)),)), dominance=dominance
        ),
        "dominant_plus_artifact": _case(
            _image(((slice(10, 90), slice(10, 90)), (slice(0, 5), slice(0, 5)))),
            dominance=dominance,
        ),
        "exact_boundary": _case(
            _image(((slice(5, 65), slice(5, 65)), (slice(75, 95), slice(70, 90)))),
            dominance=dominance,
        ),
        "below_boundary": _case(
            _image(((slice(5, 65), slice(5, 65)), (slice(75, 96), slice(70, 90)))),
            dominance=dominance,
        ),
        "equal_components": _case(
            _image(((slice(5, 35), slice(5, 35)), (slice(60, 90), slice(60, 90)))),
            dominance=dominance,
        ),
    }
    checks = {
        "legacy_single_pixelwise_unchanged": cases["legacy_single"]["masks_equal"] is True,
        "dominant_artifact_resolved": cases["dominant_plus_artifact"]["amended_status"]
        == FOVStatus.VALID.value,
        "exact_boundary_inclusive": cases["exact_boundary"]["amended_status"]
        == FOVStatus.VALID.value,
        "below_boundary_fail_closed": cases["below_boundary"]["amended_status"]
        == FOVStatus.MULTIPLE_COMPONENTS.value,
        "equal_components_fail_closed": cases["equal_components"]["amended_status"]
        == FOVStatus.MULTIPLE_COMPONENTS.value,
    }
    payload = {
        "schema_version": 1,
        "status": "RPS_MED_R1_V0_7_SYNTHETIC_FOV_AMENDMENT_COMPLETE",
        "dominant_component_min_share": dominance,
        "cases": cases,
        "checks": checks,
        "scientific_data_opened": False,
        "raw_input_bytes_opened": False,
        "relations_computed": False,
        "formal_r1": "NOT_RUN_SYNTHETIC_ONLY",
        "gate0": "NOT_RERUN",
        "training": "FORBIDDEN_NOT_RUN",
        "local_paths_recorded": False,
        "source_identifiers_recorded": False,
    }
    if not all(checks.values()):
        raise RuntimeError("v0.7 synthetic FOV amendment checks failed")
    _write_once(args.output, payload)
    print(json.dumps({"status": payload["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the write-once DDR-train FOV invariance audit for RPS-Med R1 v0.7."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from nts_med.reproducibility.freeze import sha256_file
from rps_med.cli.formal_r1 import _mapping, _write_once
from rps_med.config_v07 import load_v07_config
from rps_med.source_io_v07 import audit_ddr_train_fov_v07


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ddr-root", type=Path, required=True)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError("v0.7 development FOV audit output already exists")
    root = args.repository_root.resolve()
    config = load_v07_config(root, args.config)
    identity = _mapping(config.get("input_identity"), name="input_identity")
    if sha256_file(args.private_manifest) != identity.get("ddr_private_manifest"):
        raise ValueError("DDR private manifest identity mismatch")
    development = _mapping(config.get("development_validation"), name="development_validation")
    amendment = _mapping(config.get("measurement_amendment"), name="measurement_amendment")
    audit = audit_ddr_train_fov_v07(
        args.ddr_root,
        dominant_component_min_share=float(amendment["dominant_component_min_share"]),
    )
    checks = {
        "input_count": audit["observed_input_count"] == development["expected_input_count"],
        "decode_exclusions": audit["image_decode_exclusion_count"]
        == development["image_decode_exclusion_count_required"],
        "legacy_mask_invariance": audit["legacy_valid_mask_difference_count"]
        == development["legacy_valid_mask_difference_count_required"],
        "lesion_masks_not_opened": audit["lesion_masks_opened"] is False,
        "relations_not_computed": audit["relations_computed"] is False,
    }
    status = (
        "RPS_MED_R1_V0_7_DEVELOPMENT_FOV_AUDIT_COMPLETE"
        if all(checks.values())
        else "RPS_MED_R1_V0_7_DEVELOPMENT_FOV_AUDIT_BLOCKED"
    )
    payload = {
        "schema_version": 1,
        "status": status,
        "analysis_role": "DEVELOPMENT_FOV_INVARIANCE_ONLY",
        "checks": checks,
        "audit": audit,
        "held_out_or_test_scientific_access": False,
        "lesion_masks_opened": False,
        "relations_computed": False,
        "formal_r1": "NOT_RUN",
        "gate0": "NOT_RERUN",
        "training": "FORBIDDEN_NOT_RUN",
        "local_paths_recorded": False,
        "source_identifiers_recorded": False,
    }
    _write_once(args.output, payload)
    print(json.dumps({"status": status}))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

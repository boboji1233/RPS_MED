from __future__ import annotations

import csv
import json
from pathlib import Path

from rps_med.stage3a_reporting import write_stage3a_csv_reports


def _summary(value: float, *, family: str = "pair") -> dict[str, object]:
    scalar = {"mean": value, "sample_sd": 0.01, "minimum": value, "maximum": value}
    return {
        "aliases": ["alias"],
        "tier": "P0",
        "family": family,
        "metrics": {
            "spectrum_macro_f1": scalar,
            "relation_accuracy": scalar,
            "phase_total_variation": scalar,
            "transition_auc": scalar,
            "critical_log2_mae": scalar,
            "pair_swap_consistency": scalar,
            "invalid_spectrum_rate": scalar,
            "multi_transition_rate": scalar,
            "per_state_f1": {
                state: scalar
                for state in (
                    "LESS",
                    "GREATER",
                    "EQUIVALENT",
                    "STRUCTURAL_INCOMPARABLE",
                    "UNRESOLVED",
                    "MERGED_SI_UNRESOLVED",
                )
            },
        },
        "resource": {
            "total_pipeline_runtime_seconds": scalar,
            "peak_cuda_vram_bytes": scalar,
            "total_pipeline_parameter_count": scalar,
            "best_encoder_epoch": scalar,
            "best_head_epoch": scalar,
        },
    }


def test_stage3a_reporting_expands_aliases_and_preserves_unadaptable_rows(
    tmp_path: Path,
) -> None:
    canonical_ids = {
        "B0",
        "B1",
        "B2_A1",
        "B3_A3",
        "B4",
        "B5",
        "B6_A0_A12full",
        "B7",
        "A2",
        "A4",
        "A5",
        "A6",
        "A7",
        "A8",
        "A9",
        "A10",
        "A11",
        "A12_signed",
        "A12_absolute",
        "A14",
        "A15",
        "A16",
    }
    analysis = {
        "status": "STAGE3A_NODE_LEVEL_ANALYSIS_COMPLETE",
        "official_test_accessed": False,
        "ddr_accessed": False,
        "evidence_level": "DEVELOPMENT_ONLY_SUPPORTIVE_EVIDENCE",
        "seed_summaries": {run_id: _summary(0.5) for run_id in canonical_ids},
    }
    analysis["seed_summaries"]["B0"]["resource"]["best_encoder_epoch"] = None
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    baseline, ablation, resource = write_stage3a_csv_reports(
        analysis_path, tmp_path / "out"
    )

    with baseline.open(encoding="utf-8", newline="") as handle:
        baseline_rows = {row["baseline_id"]: row for row in csv.DictReader(handle)}
    assert baseline_rows["B6"]["canonical_run_id"] == "B6_A0_A12full"
    assert baseline_rows["B8"]["status"] == "NOT_FAIRLY_ADAPTABLE"
    assert baseline_rows["B8"]["macro_f1_mean"] == ""

    with ablation.open(encoding="utf-8", newline="") as handle:
        ablation_rows = {row["ablation_id"]: row for row in csv.DictReader(handle)}
    assert ablation_rows["A0"]["canonical_run_id"] == "B6_A0_A12full"
    assert ablation_rows["A11"]["five_state_primary_comparable"] == "NO"
    assert resource.exists()


def test_stage3a_reporting_rejects_boundary_mismatch(tmp_path: Path) -> None:
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(
        json.dumps(
            {
                "status": "STAGE3A_NODE_LEVEL_ANALYSIS_COMPLETE",
                "official_test_accessed": True,
                "ddr_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    try:
        write_stage3a_csv_reports(analysis_path, tmp_path / "out")
    except ValueError as error:
        assert "official-test boundary" in str(error)
    else:
        raise AssertionError("boundary mismatch must fail closed")

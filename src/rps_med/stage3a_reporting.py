"""Deterministic tracked-table reporting for RPS-Med Stage 3A."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

_BASELINES: tuple[tuple[str, str, str, str], ...] = (
    ("B0", "B0", "Scale-specific majority", "EXECUTED"),
    ("B1", "B1", "Analytic image-burden pipeline", "EXECUTED"),
    ("B2", "B2_A1", "Frozen-ImageNet direct pair", "EXECUTED"),
    ("B3", "B3_A3", "Spectrum-only pair model", "EXECUTED"),
    ("B4", "B4", "Lambda-conditioned five-state classifier", "EXECUTED"),
    ("B5", "B5", "Independent-per-scale heads", "EXECUTED"),
    ("B6", "B6_A0_A12full", "Full RPS-Med", "EXECUTED"),
    ("B7", "B7", "RankNet-style scalar stress test", "EXECUTED_STRESS_TEST"),
    ("B8", "", "ConPrO-style adaptation", "NOT_FAIRLY_ADAPTABLE"),
    (
        "B9",
        "",
        "Noisy ordinal progression adaptation",
        "NOT_FAIRLY_ADAPTABLE",
    ),
)

_ABLATIONS: tuple[tuple[str, str, str], ...] = (
    ("A0", "B6_A0_A12full", "Full model"),
    ("A1", "B2_A1", "Spectrum only with frozen ImageNet encoder"),
    ("A2", "A2", "Spectrum plus burden"),
    ("A3", "B3_A3", "Spectrum plus burden plus presence"),
    ("A4", "A4", "Spectrum plus transition"),
    ("A5", "A5", "Spectrum plus critical"),
    ("A6", "A6", "Spectrum plus transition plus critical"),
    ("A7", "A7", "Full minus burden supervision"),
    ("A8", "A8", "Full minus transition loss"),
    ("A9", "A9", "Full minus critical loss"),
    ("A10", "A10", "Full minus lesion-presence supervision"),
    ("A11", "A11", "Merge SI and UNRESOLVED; four-state diagnostic"),
    ("A12-signed", "A12_signed", "Signed difference only"),
    ("A12-absolute", "A12_absolute", "Absolute difference only"),
    (
        "A12-full",
        "B6_A0_A12full",
        "Signed plus absolute difference plus product",
    ),
    ("A13-frozen", "A6", "Frozen encoder"),
    ("A13-finetuned", "B6_A0_A12full", "Fine-tuned encoder"),
    ("A14", "A14", "No class weighting"),
    ("A15", "A15", "No geometric augmentation"),
    ("A16", "A16", "Soft at-most-one-transition regularizer"),
)

_METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("spectrum_macro_f1", "macro_f1"),
    ("relation_accuracy", "accuracy"),
    ("phase_total_variation", "phase_tv"),
    ("transition_auc", "transition_auc"),
    ("critical_log2_mae", "critical_log2_mae"),
    ("pair_swap_consistency", "pair_swap_consistency"),
    ("invalid_spectrum_rate", "invalid_spectrum_rate"),
    ("multi_transition_rate", "multi_transition_rate"),
)


def _load_analysis(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "STAGE3A_NODE_LEVEL_ANALYSIS_COMPLETE":
        raise ValueError("Stage 3A analysis is not complete")
    if payload.get("official_test_accessed") is not False:
        raise ValueError("Stage 3A analysis does not attest the official-test boundary")
    if payload.get("ddr_accessed") is not False:
        raise ValueError("Stage 3A analysis does not attest the DDR boundary")
    return cast(Mapping[str, Any], payload)


def _metric_fields(summary: Mapping[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    metrics = summary["metrics"]
    for source, target in _METRIC_COLUMNS:
        value = metrics[source]
        fields[f"{target}_mean"] = "" if value is None else repr(value["mean"])
        fields[f"{target}_sample_sd"] = (
            "" if value is None else repr(value["sample_sd"])
        )
    for state in (
        "LESS",
        "GREATER",
        "EQUIVALENT",
        "STRUCTURAL_INCOMPARABLE",
        "UNRESOLVED",
    ):
        value = metrics["per_state_f1"].get(state)
        prefix = f"f1_{state.lower()}"
        fields[f"{prefix}_mean"] = "" if value is None else repr(value["mean"])
        fields[f"{prefix}_sample_sd"] = (
            "" if value is None else repr(value["sample_sd"])
        )
    return fields


def _empty_metric_fields() -> dict[str, str]:
    fields: dict[str, str] = {}
    for _, target in _METRIC_COLUMNS:
        fields[f"{target}_mean"] = ""
        fields[f"{target}_sample_sd"] = ""
    for state in (
        "less",
        "greater",
        "equivalent",
        "structural_incomparable",
        "unresolved",
    ):
        fields[f"f1_{state}_mean"] = ""
        fields[f"f1_{state}_sample_sd"] = ""
    return fields


def _resource_value(
    resource: Mapping[str, Any], name: str, statistic: str
) -> str:
    summary = resource.get(name)
    if summary is None:
        return ""
    return repr(summary[statistic])


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Stage 3A report table is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_stage3a_csv_reports(
    analysis_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Write baseline, ablation, and resource tables from one frozen analysis."""

    analysis = _load_analysis(analysis_path)
    summaries = analysis["seed_summaries"]
    evidence = str(analysis["evidence_level"])
    baseline_rows: list[dict[str, Any]] = []
    for public_id, canonical_id, name, status in _BASELINES:
        row: dict[str, Any] = {
            "baseline_id": public_id,
            "canonical_run_id": canonical_id,
            "method": name,
            "status": status,
            "evidence_level": evidence,
            "seed_count": 3 if canonical_id else 0,
            "five_state_native": (
                "NO"
                if public_id in {"B7", "B8", "B9"}
                else "YES"
            ),
        }
        if canonical_id:
            row.update(_metric_fields(summaries[canonical_id]))
        else:
            row.update(_empty_metric_fields())
        row["adaptation_note"] = {
            "B7": (
                "Representational stress test; native scalar output cannot express "
                "STRUCTURAL_INCOMPARABLE or UNRESOLVED."
            ),
            "B8": (
                "Primary method learns ordinal severity representations from normal "
                "anchors; its public implementation is under development and does not "
                "define the frozen five-state, 17-scale pair task."
            ),
            "B9": (
                "Primary method uses longitudinal OCT better/stable/worse plus an "
                "ungradable other class; mapping other to SI or UNRESOLVED would change "
                "its semantics."
            ),
        }.get(public_id, "")
        baseline_rows.append(row)

    ablation_rows: list[dict[str, Any]] = []
    for public_id, canonical_id, name in _ABLATIONS:
        summary = summaries[canonical_id]
        row = {
            "ablation_id": public_id,
            "canonical_run_id": canonical_id,
            "variant": name,
            "status": "EXECUTED",
            "evidence_level": evidence,
            "seed_count": 3,
            "five_state_primary_comparable": "NO" if public_id == "A11" else "YES",
            **_metric_fields(summary),
        }
        ablation_rows.append(row)

    resource_rows: list[dict[str, Any]] = []
    for run_id, summary in sorted(summaries.items()):
        resource = summary["resource"]
        resource_rows.append(
            {
                "canonical_run_id": run_id,
                "aliases": ";".join(summary["aliases"]),
                "tier": summary["tier"],
                "family": summary["family"],
                "seed_count": 3,
                "runtime_seconds_mean": _resource_value(
                    resource, "total_pipeline_runtime_seconds", "mean"
                ),
                "runtime_seconds_sample_sd": _resource_value(
                    resource, "total_pipeline_runtime_seconds", "sample_sd"
                ),
                "peak_cuda_vram_bytes_max": _resource_value(
                    resource, "peak_cuda_vram_bytes", "maximum"
                ),
                "parameter_count": _resource_value(
                    resource, "total_pipeline_parameter_count", "mean"
                ),
                "best_encoder_epoch_mean": _resource_value(
                    resource, "best_encoder_epoch", "mean"
                ),
                "best_head_epoch_mean": _resource_value(
                    resource, "best_head_epoch", "mean"
                ),
                "evidence_level": evidence,
            }
        )

    output = Path(output_dir)
    baseline_path = output / "RPS_MED_STAGE3A_BASELINE_RESULTS_V1.csv"
    ablation_path = output / "RPS_MED_STAGE3A_ABLATION_RESULTS_V1.csv"
    resource_path = output / "RPS_MED_STAGE3A_RESOURCE_REPORT_V1.csv"
    _write_csv(baseline_path, baseline_rows)
    _write_csv(ablation_path, ablation_rows)
    _write_csv(resource_path, resource_rows)
    return baseline_path, ablation_path, resource_path


__all__ = ["write_stage3a_csv_reports"]

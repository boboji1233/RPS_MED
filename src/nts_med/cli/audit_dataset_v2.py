"""Prompt 2 dataset qualification audit with fail-closed scientific status."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from nts_med.cli._common import print_status, software_versions, write_json
from nts_med.data.adapters.base import AdapterStatus, DatasetSample
from nts_med.data.adapters.registry import create_adapter
from nts_med.data.audit import audit_manifest
from nts_med.data.config import DatasetConfig, DatasetConfigError, load_dataset_config
from nts_med.data.dedup import (
    FingerprintMethod,
    ImageFingerprint,
    find_cross_split_duplicate_candidates,
    fingerprint_image,
    group_fingerprints,
)
from nts_med.data.manifests import build_manifest, sha256_file, write_manifest
from nts_med.data.semantic_audit import AuditStageStatus, SemanticAudit, audit_samples


def build_parser() -> argparse.ArgumentParser:
    """Return the parser for a local, non-training dataset qualification audit."""

    parser = argparse.ArgumentParser(
        description=(
            "Audit an authorized local dataset through its source-specific adapter; "
            "never downloads data or produces a scientific Gate 0 decision."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="configs/data/*.yaml")
    parser.add_argument("--root", type=Path, default=None, help="explicit authorized root")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="separate authorized image root, required by annotation-only datasets",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=None,
        help="separate official metadata root, required by MAPLES-DR",
    )
    parser.add_argument("--output", type=Path, required=True, help="local audit directory")
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="local relative-path file manifest CSV (default: OUTPUT/file_manifest.csv)",
    )
    parser.add_argument(
        "--skip-image-metadata",
        action="store_true",
        help="hash files without decoding image dimensions/channels",
    )
    parser.add_argument(
        "--skip-fingerprints",
        action="store_true",
        help="skip decoded-pixel/dHash/pHash duplicate candidate audit",
    )
    parser.add_argument(
        "--dhash-max-distance",
        type=int,
        default=4,
        help="candidate-only dHash Hamming threshold",
    )
    parser.add_argument(
        "--phash-max-distance",
        type=int,
        default=6,
        help="candidate-only pHash Hamming threshold",
    )
    return parser


def _configured_root(config: DatasetConfig, explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        return explicit_root
    env_value = os.environ.get(config.root_env_var)
    if env_value:
        return Path(env_value)
    return Path("data") / "raw" / config.name


def _configured_image_root(
    config: DatasetConfig, explicit_root: Path | None
) -> Path | None:
    if explicit_root is not None:
        return explicit_root
    env_name = config.image_root_env_var
    if env_name:
        env_value = os.environ.get(env_name)
        if env_value:
            return Path(env_value)
    return None


def _configured_metadata_root(
    config: DatasetConfig, explicit_root: Path | None
) -> Path | None:
    if explicit_root is not None:
        return explicit_root
    env_name = config.additional_metadata_root_env_var
    if env_name:
        env_value = os.environ.get(env_name)
        if env_value:
            return Path(env_value)
    return None


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_output_locations(
    root: Path,
    image_root: Path | None,
    metadata_root: Path | None,
    output: Path,
    manifest_output: Path,
) -> None:
    resolved_outputs = (output.resolve(strict=False), manifest_output.resolve(strict=False))
    resolved_roots = [root.resolve(strict=False)]
    if image_root is not None:
        resolved_roots.append(image_root.resolve(strict=False))
    if metadata_root is not None:
        resolved_roots.append(metadata_root.resolve(strict=False))
    for destination in resolved_outputs:
        if any(_is_within(destination, source) for source in resolved_roots):
            raise ValueError("audit outputs must remain outside all dataset roots")


def _extension_counts(records: Sequence[Any]) -> dict[str, int]:
    counts = Counter(record.suffix or "[no_suffix]" for record in records)
    return dict(sorted(counts.items()))


def _safe_adapter_facts(adapter: object) -> Mapping[str, object]:
    provider = getattr(adapter, "audit_facts", None)
    if not callable(provider):
        return {}
    facts = provider()
    if not isinstance(facts, Mapping) or not all(isinstance(key, str) for key in facts):
        raise ValueError("adapter audit_facts must return a string-keyed mapping")
    return facts


def _audit_failure(code: str, error: Exception) -> dict[str, object]:
    """Return a stable failure payload without paths or source identifiers."""

    return {
        "status": "AUDIT_FAILED",
        "error_code": code,
        "error_type": type(error).__name__,
        "software_versions": software_versions(),
    }


def _overall_audit_status(
    discovery_status: AdapterStatus,
    validation_status: AdapterStatus,
    semantic_status: AuditStageStatus | None,
    duplicate_status: object,
) -> str:
    if discovery_status is AdapterStatus.DATA_NOT_AVAILABLE:
        return "DATA_NOT_AVAILABLE"
    if validation_status is AdapterStatus.INCOMPLETE:
        return "INCOMPLETE"
    if (
        validation_status is AdapterStatus.INVALID
        or semantic_status is AuditStageStatus.ISSUES_FOUND
        or duplicate_status == AuditStageStatus.ISSUES_FOUND.value
    ):
        return "ISSUES_FOUND"
    if semantic_status is None or duplicate_status != AuditStageStatus.COMPLETE.value:
        return "INCOMPLETE"
    return "QUALIFICATION_AUDIT_COMPLETE"


def _duplicate_group_id(method: FingerprintMethod, fingerprint: str) -> str:
    digest = hashlib.sha256(
        f"nts-med-duplicate-v1\0{method.value}\0{fingerprint}".encode("ascii")
    ).hexdigest()
    return f"dup_{digest[:24]}"


def _fingerprint_audit(
    samples: Sequence[DatasetSample],
    *,
    dhash_max_distance: int,
    phash_max_distance: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    fingerprints: list[ImageFingerprint] = []
    split_by_instance: dict[str, str] = {}
    unreadable = 0
    for sample in sorted(samples, key=lambda value: value.sample_id):
        try:
            fingerprints.append(fingerprint_image(sample.sample_id, sample.image_path))
            split = getattr(sample, "official_split", None)
            split_by_instance[sample.sample_id] = (
                split if isinstance(split, str) and split else "UNKNOWN"
            )
        except (OSError, SyntaxError, ValueError):
            unreadable += 1

    duplicate_rows: list[dict[str, object]] = []
    group_counts: dict[str, int] = {}
    duplicated_instance_counts: dict[str, int] = {}
    for method in FingerprintMethod:
        groups = group_fingerprints(fingerprints, method)
        group_counts[method.value] = len(groups)
        duplicated_instance_counts[method.value] = sum(len(group.instance_ids) for group in groups)
        for group in groups:
            group_id = _duplicate_group_id(method, group.fingerprint)
            for instance_id in group.instance_ids:
                duplicate_rows.append(
                    {
                        "duplicate_group_id": group_id,
                        "method": method.value,
                        "instance_id": instance_id,
                        "official_split": split_by_instance[instance_id],
                    }
                )

    candidates = find_cross_split_duplicate_candidates(
        fingerprints,
        split_by_instance,
        dhash_max_distance=dhash_max_distance,
        phash_max_distance=phash_max_distance,
    )
    candidate_rows: list[dict[str, object]] = [
        {
            "left_id": candidate.left_id,
            "right_id": candidate.right_id,
            "left_split": candidate.left_split,
            "right_split": candidate.right_split,
            "matched_methods": ";".join(method.value for method in candidate.matched_methods),
            "dhash_distance": (
                "" if candidate.dhash_distance is None else candidate.dhash_distance
            ),
            "phash_distance": (
                "" if candidate.phash_distance is None else candidate.phash_distance
            ),
            "review_decision": "",
        }
        for candidate in candidates
    ]
    summary: dict[str, object] = {
        "status": (
            AuditStageStatus.ISSUES_FOUND.value
            if candidates or unreadable
            else AuditStageStatus.COMPLETE.value
        ),
        "fingerprinted_instances": len(fingerprints),
        "unreadable_instances": unreadable,
        "exact_group_counts": group_counts,
        "duplicated_instance_counts": duplicated_instance_counts,
        "cross_split_candidate_count": len(candidates),
        "dhash_max_distance": dhash_max_distance,
        "phash_max_distance": phash_max_distance,
        "frozen_feature_similarity": "NOT_RUN_NO_APPROVED_LOCAL_WEIGHTS",
        "automatic_deletion": False,
    }
    return summary, duplicate_rows, candidate_rows


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_mask_statistics(output: Path, semantic: SemanticAudit | None) -> None:
    rows = [] if semantic is None else [axis.to_dict() for axis in semantic.axes]
    _write_csv(
        output,
        [
            "axis",
            "expected_instances",
            "masks_present",
            "matching_rate",
            "positive_instances",
            "empty_instances",
            "unreadable_masks",
            "invalid_value_masks",
            "size_mismatch_masks",
        ],
        rows,
    )


def _write_issue_counts(output: Path, semantic: SemanticAudit | None) -> None:
    rows = (
        []
        if semantic is None
        else [
            {"issue_code": code, "count": count}
            for code, count in semantic.issue_counts.items()
        ]
    )
    _write_csv(output, ["issue_code", "count"], rows)


def _dataset_card(
    config: DatasetConfig,
    overall_status: str,
    adapter_status: str,
    manifest_hash: str | None,
    gate0_blockers: Sequence[str],
) -> str:
    blockers = "\n".join(f"- {blocker}" for blocker in gate0_blockers)
    return (
        f"# {config.name} local qualification card\n\n"
        f"- Audit status: `{overall_status}`\n"
        f"- Adapter status: `{adapter_status}`\n"
        f"- Configured layout/version profile: `{config.version}`\n"
        f"- Official source: {config.official_url}\n"
        f"- File manifest SHA256: `{manifest_hash or 'NOT_AVAILABLE'}`\n"
        "- Scientific Gate 0: `NOT_RUN`\n"
        "- Raw data redistribution: prohibited by repository policy\n\n"
        "## Gate 0 blockers\n\n"
        f"{blockers or '- None recorded (this still does not constitute Gate 0 approval).'}\n\n"
        "This is a local data-qualification artifact, not a scientific experiment "
        "or clinical claim.\n"
    )


def _split_audit_status(
    adapter_status: AdapterStatus,
    adapter_issues: Sequence[str],
    duplicate_status: object,
) -> str:
    """Return a fail-closed split status from completed prerequisite audits."""

    if adapter_status is not AdapterStatus.READY:
        return "NOT_RUN_ADAPTER_NOT_READY"
    task_split_conflict = any(
        issue.startswith("DDR_TASK_SPLIT_ASSIGNMENT_CONFLICT:")
        for issue in adapter_issues
    )
    duplicates_pending = duplicate_status == AuditStageStatus.ISSUES_FOUND.value
    if task_split_conflict and duplicates_pending:
        return "BLOCKED_PENDING_TASK_SPLIT_AND_DUPLICATE_REVIEW"
    if task_split_conflict:
        return "BLOCKED_PENDING_TASK_SPLIT_RECONCILIATION"
    if duplicates_pending:
        return "BLOCKED_PENDING_DUPLICATE_REVIEW"
    if duplicate_status == "NOT_RUN_USER_SKIPPED":
        return "BLOCKED_DUPLICATE_AUDIT_NOT_RUN"
    return "NOT_RUN_SPLIT_POLICY_NOT_FROZEN"


def main(argv: Sequence[str] | None = None) -> int:
    """Audit one configured dataset and emit path-safe aggregate reports."""

    args = build_parser().parse_args(argv)
    try:
        config = load_dataset_config(args.config)
        root = _configured_root(config, args.root)
        image_root = _configured_image_root(config, args.image_root)
        metadata_root = _configured_metadata_root(config, args.metadata_root)
        manifest_output = args.manifest_output or (args.output / "file_manifest.csv")
        _validate_output_locations(
            root,
            image_root,
            metadata_root,
            args.output,
            manifest_output,
        )
        adapter = create_adapter(
            config.name,
            root,
            image_root=image_root,
            metadata_root=metadata_root,
            release_profile=config.additional_metadata_profile,
            component_acquisition_records=config.component_acquisition_records,
        )
        discovery = adapter.discover()
        validation = adapter.validate()
    except (DatasetConfigError, OSError, TypeError, ValueError) as error:
        failure = _audit_failure("AUDIT_SETUP_FAILED", error)
        print_status(failure)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    file_summary: dict[str, object]
    manifest_hash: str | None = None
    if root.is_dir():
        try:
            records = build_manifest(
                root,
                include_image_metadata=not args.skip_image_metadata,
            )
            write_manifest(records, manifest_output)
            manifest_hash = sha256_file(manifest_output)
            integrity = audit_manifest(records)
            file_summary = {
                "status": integrity.status.value,
                "file_count": integrity.file_count,
                "total_bytes": integrity.total_bytes,
                "corrupt_file_count": len(integrity.corrupt_files),
                "unreadable_file_count": len(integrity.unreadable_files),
                "extension_counts": _extension_counts(records),
                "manifest_sha256": manifest_hash,
            }
        except (OSError, TypeError, ValueError) as error:
            failure = _audit_failure("FILE_AUDIT_FAILED", error)
            print_status(failure)
            return 2
    else:
        file_summary = {
            "status": "DATA_NOT_AVAILABLE",
            "file_count": None,
            "total_bytes": None,
            "corrupt_file_count": None,
            "unreadable_file_count": None,
            "extension_counts": {},
            "manifest_sha256": None,
        }

    samples: tuple[DatasetSample, ...] = ()
    semantic: SemanticAudit | None = None
    duplicate_summary: dict[str, object] = {
        "status": "NOT_RUN_ADAPTER_NOT_READY",
        "frozen_feature_similarity": "NOT_RUN_NO_APPROVED_LOCAL_WEIGHTS",
        "automatic_deletion": False,
    }
    duplicate_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    if validation.status is AdapterStatus.READY:
        try:
            samples = tuple(adapter.samples())
            semantic = audit_samples(
                samples,
                axes=("ma", "hem", "ex", "cws"),
                positive_class=config.positive_class,
            )
            if args.skip_fingerprints:
                duplicate_summary = {
                    "status": "NOT_RUN_USER_SKIPPED",
                    "frozen_feature_similarity": "NOT_RUN_NO_APPROVED_LOCAL_WEIGHTS",
                    "automatic_deletion": False,
                }
            else:
                duplicate_summary, duplicate_rows, candidate_rows = _fingerprint_audit(
                    samples,
                    dhash_max_distance=args.dhash_max_distance,
                    phash_max_distance=args.phash_max_distance,
                )
        except (OSError, TypeError, ValueError) as error:
            failure = _audit_failure("SEMANTIC_OR_DUPLICATE_AUDIT_FAILED", error)
            print_status(failure)
            return 2

    try:
        adapter_facts = _safe_adapter_facts(adapter)
    except (OSError, TypeError, ValueError) as error:
        failure = _audit_failure("ADAPTER_FACT_AUDIT_FAILED", error)
        print_status(failure)
        return 2

    duplicate_status = duplicate_summary.get("status")
    adapter_status = validation.status.value
    overall_status = _overall_audit_status(
        discovery.status,
        validation.status,
        None if semantic is None else semantic.status,
        duplicate_status,
    )
    task_split_conflict = any(
        issue.startswith("DDR_TASK_SPLIT_ASSIGNMENT_CONFLICT:")
        for issue in validation.issues
    )
    gate0_blockers = [
        "At least two independent, fully licensed four-axis expert-mask sources are required.",
        "Thresholds, intervals, FOV policy, and split policy are not yet calibrated and frozen.",
    ]
    if duplicate_status == AuditStageStatus.ISSUES_FOUND.value:
        gate0_blockers.append(
            "Duplicate candidates require review before a leakage-safe split can be sealed."
        )
    elif duplicate_status == "NOT_RUN_USER_SKIPPED":
        gate0_blockers.append(
            "The duplicate audit was skipped, so a leakage-safe split cannot be sealed."
        )
    if task_split_conflict:
        gate0_blockers.append(
            "DDR grading and lesion task split assignments conflict and require reconciliation."
        )
    scientific_decision = config.license_record.get("scientific_claim_decision")
    configured_gate0 = config.license_record.get("gate0_decision")
    if isinstance(scientific_decision, str) and scientific_decision.startswith("PENDING"):
        gate0_blockers.append("Dataset license scope still requires confirmation for publication.")
    source_identity = config.license_record.get("source_release_identity")
    if isinstance(source_identity, str) and source_identity.startswith(
        ("UNVERIFIED", "PARTIAL")
    ):
        gate0_blockers.append(
            "The local copy's official source release and archive checksum are not locked."
        )
    label_semantics = config.license_record.get("label_semantics_status")
    if isinstance(label_semantics, str) and label_semantics.startswith("REVIEW_REQUIRED"):
        gate0_blockers.append(
            "Configured dataset label semantics require authoritative confirmation."
        )
    if isinstance(configured_gate0, str) and configured_gate0.startswith("BLOCKED"):
        gate0_blockers.append(configured_gate0)
    if validation.status is not AdapterStatus.READY:
        gate0_blockers.append(f"Adapter is {validation.status.value}, not READY.")

    split_status = _split_audit_status(
        validation.status,
        validation.issues,
        duplicate_status,
    )

    payload: dict[str, object] = {
        "status": overall_status,
        "dataset": config.name,
        "configured_version": config.version,
        "official_url": config.official_url,
        "adapter": {
            "discovery_status": discovery.status.value,
            "validation_status": adapter_status,
            "issue_codes": list(validation.issues),
            "message": validation.message,
            "facts": dict(adapter_facts),
        },
        "file_audit": file_summary,
        "semantic_audit": None if semantic is None else semantic.to_dict(),
        "duplicate_audit": duplicate_summary,
        "split_audit": {
            "status": split_status,
            "grouping_level": (
                "instance" if config.raw.get("patient_id_available") is False else "UNRESOLVED"
            ),
            "new_split_written": False,
        },
        "pretraining_exposure_audit": "NOT_RUN_NO_MODELS_IN_SCOPE",
        "qualification": {
            "g_data": "NO_GO",
            "gate0": "NOT_RUN",
            "gate0_eligible": False,
            "blockers": gate0_blockers,
        },
        "software_versions": software_versions(),
    }

    write_json(args.output / "summary.json", payload)
    provenance = {
        "dataset_config": dict(config.raw),
        "license_record": dict(config.license_record),
        "manifest_sha256": manifest_hash,
        "root_recorded": False,
        "source_identifiers_in_provenance": False,
        "local_manifest_written": manifest_hash is not None,
        "local_manifest_contains_source_relative_paths": manifest_hash is not None,
        "local_manifest_distribution": (
            "LOCAL_ONLY_GIT_IGNORED" if manifest_hash is not None else "NOT_WRITTEN"
        ),
    }
    (args.output / "provenance.yaml").write_text(
        yaml.safe_dump(provenance, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    (args.output / "dataset_card.md").write_text(
        _dataset_card(
            config,
            overall_status,
            adapter_status,
            manifest_hash,
            gate0_blockers,
        ),
        encoding="utf-8",
        newline="\n",
    )
    _write_mask_statistics(args.output / "mask_statistics.csv", semantic)
    _write_issue_counts(args.output / "matching_issues.csv", semantic)
    _write_csv(
        args.output / "duplicate_groups.csv",
        ["duplicate_group_id", "method", "instance_id", "official_split"],
        duplicate_rows,
    )
    _write_csv(
        args.output / "duplicate_candidates.csv",
        [
            "left_id",
            "right_id",
            "left_split",
            "right_split",
            "matched_methods",
            "dhash_distance",
            "phash_distance",
            "review_decision",
        ],
        candidate_rows,
    )
    _write_csv(
        args.output / "manual_review.csv",
        [
            "left_id",
            "right_id",
            "left_split",
            "right_split",
            "matched_methods",
            "dhash_distance",
            "phash_distance",
            "review_decision",
        ],
        candidate_rows,
    )
    print_status(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

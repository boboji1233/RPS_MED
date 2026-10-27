"""Strict dataset-configuration and local license-gate helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class DatasetConfigError(ValueError):
    """Raised when a dataset configuration cannot support an auditable read."""


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Validated subset of the data configuration used by Prompt 2."""

    name: str
    version: str
    root_env_var: str
    official_url: str
    license_record_path: Path
    license_record: Mapping[str, object]
    component_acquisition_records: Mapping[str, Path]
    raw: Mapping[str, object]

    @property
    def image_root_env_var(self) -> str | None:
        value = self.raw.get("image_root_env_var")
        return value if isinstance(value, str) and value else None

    @property
    def additional_metadata_root_env_var(self) -> str | None:
        value = self.raw.get("additional_metadata_root_env_var")
        return value if isinstance(value, str) and value else None

    @property
    def additional_metadata_profile(self) -> Mapping[str, object] | None:
        value = self.raw.get("additional_metadata_profile")
        if value is None:
            return None
        if not isinstance(value, Mapping) or not all(
            isinstance(key, str) for key in value
        ):
            raise DatasetConfigError("additional_metadata_profile must be a mapping")
        return value

    @property
    def lesion_axes(self) -> tuple[str, ...]:
        value = self.raw.get("lesion_axes")
        if not isinstance(value, list) or not all(
            isinstance(axis, str) and axis for axis in value
        ):
            raise DatasetConfigError("lesion_axes must be a list of non-empty strings")
        return tuple(value)

    @property
    def positive_class(self) -> int:
        value = self.raw.get("positive_class", 255)
        if isinstance(value, bool) or not isinstance(value, int):
            raise DatasetConfigError("positive_class must be an integer")
        return value


def _required_text(config: Mapping[str, object], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetConfigError(f"dataset config requires non-empty {key}")
    if value.startswith("REQUIRED_") or value == "MANUAL_RECORD_REQUIRED":
        raise DatasetConfigError(f"dataset config field {key} is still a placeholder")
    return value


def _load_yaml_mapping(path: Path, *, kind: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise DatasetConfigError(f"cannot read {kind}: {type(exc).__name__}") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DatasetConfigError(f"{kind} must be a YAML mapping")
    return value


def _component_acquisition_records(
    raw: Mapping[str, object], repository_root: Path
) -> dict[str, Path]:
    configured = raw.get("component_acquisition_records", {})
    if not isinstance(configured, Mapping) or not all(
        isinstance(component, str)
        and bool(component.strip())
        and isinstance(reference, str)
        and bool(reference.strip())
        for component, reference in configured.items()
    ):
        raise DatasetConfigError(
            "component_acquisition_records must map component names to repository paths"
        )

    records: dict[str, Path] = {}
    for component, reference in configured.items():
        assert isinstance(component, str)
        assert isinstance(reference, str)
        record_path = (repository_root / reference).resolve(strict=False)
        try:
            record_path.relative_to(repository_root)
        except ValueError as exc:
            raise DatasetConfigError(
                "component acquisition records must remain inside the repository"
            ) from exc
        records[component] = record_path
    return records


def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Load a repository data config and its explicit license decision.

    Relative license paths are resolved from the repository root inferred from
    ``configs/data/<name>.yaml``.  Error messages deliberately omit the external
    dataset root and any source identifiers.
    """

    source = Path(path)
    raw = _load_yaml_mapping(source, kind="dataset config")
    schema_version = raw.get("schema_version")
    if schema_version != 1:
        raise DatasetConfigError("dataset config schema_version must be 1")
    name = _required_text(raw, "name")
    version = _required_text(raw, "version")
    root_env_var = _required_text(raw, "root_env_var")
    official_url = _required_text(raw, "official_url")
    license_reference = _required_text(raw, "license_record")

    try:
        repository_root = source.resolve(strict=True).parents[2]
    except (OSError, IndexError) as exc:
        raise DatasetConfigError("dataset config is not under configs/data") from exc
    license_path = (repository_root / license_reference).resolve(strict=False)
    try:
        license_path.relative_to(repository_root)
    except ValueError as exc:
        raise DatasetConfigError("license_record must remain inside the repository") from exc
    license_record = _load_yaml_mapping(license_path, kind="license record")
    if license_record.get("schema_version") != 1:
        raise DatasetConfigError("license record schema_version must be 1")
    if license_record.get("dataset") != name:
        raise DatasetConfigError("license record dataset does not match config")
    decision = license_record.get("local_audit_decision")
    if not isinstance(decision, str) or not decision:
        raise DatasetConfigError("license record requires local_audit_decision")
    if decision in {"BLOCKED", "DENIED", "NOT_REVIEWED"}:
        raise DatasetConfigError("license record does not permit local audit")
    component_records = _component_acquisition_records(raw, repository_root)

    parsed = DatasetConfig(
        name=name,
        version=version,
        root_env_var=root_env_var,
        official_url=official_url,
        license_record_path=license_path,
        license_record=license_record,
        component_acquisition_records=component_records,
        raw=raw,
    )
    # Trigger strict validation while no data have been opened.
    _ = parsed.lesion_axes
    _ = parsed.positive_class
    _ = parsed.additional_metadata_profile
    return parsed


__all__ = ["DatasetConfig", "DatasetConfigError", "load_dataset_config"]

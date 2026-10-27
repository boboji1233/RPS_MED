"""Shared, side-effect-light helpers for M0-M1 command-line tools."""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

from nts_med import __version__


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load one YAML mapping and reject empty or non-mapping documents."""

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML configuration: {source}: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"configuration must be a YAML mapping: {source}")
    if not all(isinstance(key, str) for key in loaded):
        raise ValueError(f"configuration keys must be strings: {source}")
    return loaded


def installed_version(distribution: str) -> str:
    """Return an installed distribution version without making network calls."""

    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def software_versions() -> dict[str, str]:
    """Return the minimal software snapshot attached to generated artifacts."""

    return {
        "nts_med": __version__,
        "python": platform.python_version(),
        "numpy": installed_version("numpy"),
        "pillow": installed_version("Pillow"),
        "pyyaml": installed_version("PyYAML"),
        "pyarrow": installed_version("pyarrow"),
        "xlrd": installed_version("xlrd"),
    }


def write_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    """Write deterministic UTF-8 JSON, creating only the requested parent path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination


def print_status(payload: Mapping[str, object]) -> None:
    """Print one machine-readable status line."""

    print(json.dumps(dict(payload), sort_keys=True))


def unavailable_payload(path: str | Path, *, kind: str = "data") -> dict[str, object]:
    """Build a no-count availability state that cannot be mistaken for an empty dataset."""

    return {
        "status": "DATA_NOT_AVAILABLE",
        "kind": kind,
        "path": str(path),
        "count": None,
        "software_versions": software_versions(),
    }

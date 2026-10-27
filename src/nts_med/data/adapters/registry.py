"""Explicit adapter registry; creation never performs network access."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from nts_med.data.adapters.base import DatasetAdapter
from nts_med.data.adapters.ddr import DDRAdapter
from nts_med.data.adapters.idrid import IDRiDAdapter
from nts_med.data.adapters.maples_dr import MaplesDRAdapter

_CANONICAL_NAMES = ("ddr", "idrid", "maples_dr")
_ALIASES = {
    "ddr": "ddr",
    "idrid": "idrid",
    "maples": "maples_dr",
    "maples_dr": "maples_dr",
    "maples-dr": "maples_dr",
}


def registered_adapters() -> tuple[str, ...]:
    """Return stable canonical adapter names."""

    return _CANONICAL_NAMES


def create_adapter(
    name: str,
    root: str | Path,
    *,
    image_root: str | Path | None = None,
    metadata_root: str | Path | None = None,
    release_profile: Mapping[str, object] | None = None,
    component_acquisition_records: Mapping[str, Path] | None = None,
) -> DatasetAdapter:
    """Construct a known read-only adapter without inspecting or downloading data."""

    if not isinstance(name, str):
        raise TypeError("adapter name must be a string")
    canonical = _ALIASES.get(name.casefold())
    if canonical is None:
        raise ValueError("unknown dataset adapter")
    if canonical == "ddr":
        if (
            image_root is not None
            or metadata_root is not None
            or release_profile is not None
            or component_acquisition_records
        ):
            raise ValueError("extra roots are supported only by maples_dr")
        return DDRAdapter(root)
    if canonical == "idrid":
        if image_root is not None or metadata_root is not None or release_profile is not None:
            raise ValueError("extra roots are supported only by maples_dr")
        records = component_acquisition_records or {}
        unexpected = set(records) - {"B. Disease Grading"}
        if unexpected:
            raise ValueError("unknown IDRiD component acquisition record")
        return IDRiDAdapter(
            root,
            grading_acquisition_record=records.get("B. Disease Grading"),
        )
    if component_acquisition_records:
        raise ValueError("component acquisition records are supported only by idrid")
    return MaplesDRAdapter(
        root,
        image_root=image_root,
        metadata_root=metadata_root,
        release_profile=release_profile,
    )


__all__ = ["create_adapter", "registered_adapters"]

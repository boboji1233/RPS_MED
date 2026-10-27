from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from rps_med.tjdr_io import (
    TJDR_FOV_ALGORITHM_VERSION,
    audit_tjdr_train,
    inspect_tjdr_structure,
    load_tjdr_train_measurements,
)


def _png(array: np.ndarray, *, mode: str) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(array, mode=mode).save(stream, format="PNG")
    return stream.getvalue()


def _pair(
    stem: str, *, invalid_class: bool = False, all_background: bool = False
) -> tuple[bytes, bytes]:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:32, :32]
    image[(xx - 16) ** 2 + (yy - 16) ** 2 <= 14**2] = 80
    mask = np.zeros((32, 32), dtype=np.uint8)
    if not all_background:
        mask[15:17, 15:17] = 9 if invalid_class else 3
    return _png(image, mode="RGB"), _png(mask, mode="L")


def _archive(path: Path, stems: tuple[str, ...], *, invalid_class: bool = False) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for index, stem in enumerate(stems):
            image, mask = _pair(stem, invalid_class=invalid_class and index == 0)
            archive.writestr(f"train/image/{stem}.png", image)
            archive.writestr(f"train/annotation/{stem}.png", mask)


def test_multi_part_structure_and_measurement_are_complete_and_opaque(tmp_path: Path) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    _archive(first, ("source-a",))
    _archive(second, ("source-b",))

    boundary = inspect_tjdr_structure((first, second), split="train")
    audit = audit_tjdr_train((first, second), expected_count=2)
    measurements = load_tjdr_train_measurements((first, second), expected_count=2)

    assert boundary.archive_count == 2
    assert boundary.combined_member_count == 4
    assert boundary.image_count == boundary.annotation_count == 2
    assert audit["status"] == "TJDR_TRAIN_AUDIT_PASS"
    assert audit["positive_image_counts"]["ma"] == 2
    assert audit["fov_algorithm_version"] == TJDR_FOV_ALGORITHM_VERSION
    assert audit["source_identifiers_recorded"] is False
    assert "source-a" not in str(audit)
    assert len(measurements.records) == 2
    assert measurements.measurement_exclusion_count == 0
    assert all(record.instance_id.startswith("tjdr_") for record in measurements.records)
    assert all(set(record.primary) == {"ma", "hem", "ex", "cws"} for record in measurements.records)


def test_invalid_annotation_is_not_reinterpreted_as_negative(tmp_path: Path) -> None:
    archive = tmp_path / "invalid.zip"
    _archive(archive, ("source-secret",), invalid_class=True)

    audit = audit_tjdr_train((archive,), expected_count=1)
    measurements = load_tjdr_train_measurements((archive,), expected_count=1)

    assert audit["status"] == "TJDR_TRAIN_AUDIT_BLOCKED"
    assert audit["reason_counts"] == {"ANNOTATION_UNKNOWN_CLASS": 1}
    assert "source-secret" not in str(audit)
    assert measurements.records == ()
    assert measurements.measurement_exclusion_count == 1


def test_complete_all_background_annotation_requires_explicit_policy(tmp_path: Path) -> None:
    archive = tmp_path / "background.zip"
    image, mask = _pair("background", all_background=True)
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("train/image/background.png", image)
        output.writestr("train/annotation/background.png", mask)

    blocked = audit_tjdr_train((archive,), expected_count=1)
    accepted = audit_tjdr_train(
        (archive,), expected_count=1, allow_all_background=True
    )
    measurements = load_tjdr_train_measurements(
        (archive,), expected_count=1, allow_all_background=True
    )

    assert blocked["reason_counts"] == {"ANNOTATION_NO_LESION_CLASS": 1}
    assert accepted["status"] == "TJDR_TRAIN_AUDIT_PASS"
    assert accepted["all_background_complete_annotation_count"] == 1
    assert measurements.measurement_exclusion_count == 0
    assert set(measurements.records[0].primary.values()) == {0.0}


def test_structure_rejects_cross_part_duplicate_member(tmp_path: Path) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    _archive(first, ("same",))
    _archive(second, ("same",))

    import pytest

    with pytest.raises(ValueError, match="duplicate member"):
        inspect_tjdr_structure((first, second), split="train")

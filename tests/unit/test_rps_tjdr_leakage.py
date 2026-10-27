from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from rps_med.tjdr_leakage import audit_tjdr_image_leakage


def _png(value: int) -> bytes:
    array = np.zeros((16, 16, 3), dtype=np.uint8)
    array[:, value : value + 4] = 100
    stream = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(stream, format="PNG")
    return stream.getvalue()


def _archive(path: Path, split: str, values: tuple[int, ...]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for index, value in enumerate(values):
            archive.writestr(f"{split}/image/sample-{index}.png", _png(value))
            archive.writestr(
                f"{split}/annotation/sample-{index}.png",
                _png(value),
            )


def test_leakage_audit_is_aggregate_and_detects_exact_cross_split(tmp_path: Path) -> None:
    train = tmp_path / "train.zip"
    test = tmp_path / "test.zip"
    _archive(train, "train", (1, 5))
    _archive(test, "test", (1, 9))

    result = audit_tjdr_image_leakage((train,), (test,))

    assert result["train"]["image_count"] == 2
    assert result["test"]["image_count"] == 2
    assert result["cross_split"]["exact"]["duplicate_pair_count"] == 1
    assert result["cross_split"]["exact"]["affected_train_image_count"] == 1
    assert result["cross_split"]["exact"]["affected_test_image_count"] == 1
    assert (
        result["cross_split"]["exact"]["matching_annotation_pixels_pair_count"]
        == 1
    )
    assert (
        result["cross_split"]["exact"]["conflicting_annotation_pixels_pair_count"]
        == 0
    )
    assert result["hash_rows_persisted"] is False
    assert "sample-" not in str(result)

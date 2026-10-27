from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from rps_med.model_data import (
    build_tjdr_development_cache,
    deterministic_multilabel_split,
    load_tjdr_model_cache,
    load_tjdr_test_model_arrays,
)


def _png(value: int, *, mask: bool = False) -> bytes:
    if mask:
        array = np.zeros((32, 32), dtype=np.uint8)
        if value:
            array[8:12, 8:12] = value
        image = Image.fromarray(array, mode="L")
    else:
        array = np.zeros((32, 32, 3), dtype=np.uint8)
        array[2:30, 2:30] = 20 + value
        image = Image.fromarray(array, mode="RGB")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _archive(path: Path, split: str, values: tuple[int, ...]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for index, value in enumerate(values):
            archive.writestr(f"{split}/image/sample-{index}.png", _png(value))
            archive.writestr(
                f"{split}/annotation/sample-{index}.png",
                _png((value % 4) + 1, mask=True),
            )


def test_deterministic_multilabel_split_is_reproducible() -> None:
    hashes = tuple(f"hash-{index}" for index in range(20))
    features = np.asarray(
        [[index % 2, (index // 2) % 2] for index in range(20)], dtype=np.float64
    )
    first = deterministic_multilabel_split(
        hashes, features, validation_fraction=0.2, seed=7
    )
    second = deterministic_multilabel_split(
        hashes, features, validation_fraction=0.2, seed=7
    )
    assert np.array_equal(first, second)
    assert int(first.sum()) == 4


def test_cache_excludes_test_content_and_contains_no_identifiers(tmp_path: Path) -> None:
    train = tmp_path / "train.zip"
    test = tmp_path / "test.zip"
    _archive(train, "train", tuple(range(12)))
    _archive(test, "test", (0, 20))
    cache = tmp_path / "cache.npz"

    summary = build_tjdr_development_cache(
        (train,),
        (test,),
        cache,
        image_size=224,
        validation_fraction=0.2,
        split_seed=3,
    )
    values = load_tjdr_model_cache(cache)

    assert summary["cross_test_exact_excluded_train_count"] == 1
    assert summary["unique_development_image_count"] == 11
    assert values["images"].shape == (11, 224, 224, 3)
    assert values["burdens"].shape == (11, 4)
    assert "sample-" not in str(summary)


def test_test_arrays_are_in_memory_and_aggregate_only(tmp_path: Path) -> None:
    test = tmp_path / "test.zip"
    _archive(test, "test", tuple(range(10)))

    images, burdens, summary = load_tjdr_test_model_arrays(
        (test,), expected_count=10, image_size=224
    )

    assert images.shape == (10, 224, 224, 3)
    assert burdens.shape == (10, 4)
    assert summary["test_cache_persisted"] is False
    assert "sample-" not in str(summary)

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from nts_med.reproducibility.freeze import sha256_file
from rps_med.cli.tjdr_development import main


def _png(array: np.ndarray, mode: str) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(array, mode=mode).save(stream, format="PNG")
    return stream.getvalue()


def _archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for index, source_class in enumerate((1, 2, 3, 4)):
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            yy, xx = np.ogrid[:32, :32]
            image[(xx - 16) ** 2 + (yy - 16) ** 2 <= 14**2] = 80
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[12 + index : 14 + index, 15:17] = source_class
            archive.writestr(f"train/image/sample-{index}.png", _png(image, "RGB"))
            archive.writestr(f"train/annotation/sample-{index}.png", _png(mask, "L"))


def _config(path: Path, archive: Path) -> None:
    value = yaml.safe_load(
        Path("configs/experiment/rps_med_tjdr_development_v0.1.yaml").read_text(
            encoding="utf-8"
        )
    )
    value["source"]["expected_development_count"] = 4
    value["source"]["train_archive_sha256"] = [sha256_file(archive)]
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_cli_audit_then_development_analysis_is_write_once_and_pair_free(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "train.zip"
    config = tmp_path / "config.yaml"
    audit_output = tmp_path / "audit.json"
    analysis_output = tmp_path / "analysis.json"
    _archive(archive)
    _config(config, archive)

    assert (
        main(
            [
                "--mode",
                "audit",
                "--config",
                str(config),
                "--train-archive",
                str(archive),
                "--output",
                str(audit_output),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--mode",
                "analyse",
                "--config",
                str(config),
                "--train-archive",
                str(archive),
                "--audit-lock",
                str(audit_output),
                "--output",
                str(analysis_output),
            ]
        )
        == 0
    )
    payload = json.loads(analysis_output.read_text(encoding="utf-8"))
    assert payload["status"] == "TJDR_TRAIN_RELATION_PERSISTENCE_DEVELOPMENT_COMPLETE"
    assert payload["source"]["pair_count"] == 6
    assert payload["pair_rows_persisted"] is False
    assert payload["test_scientific_access"] is False

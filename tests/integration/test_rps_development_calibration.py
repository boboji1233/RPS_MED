from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from rps_med.cli.development_calibration import main
from rps_med.source_io import (
    load_ddr_confirmation_measurements,
    load_maples_confirmation_measurements,
    validate_ddr_confirmation_manifest,
)


def _png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _maples(main_path: Path, additional_path: Path) -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:12, 10:12] = 1
    with zipfile.ZipFile(main_path, "w") as archive:
        for index in range(4):
            key = f"train_{index}"
            for directory in (
                "Microaneurysms",
                "Hemorrhages",
                "Exudates",
                "CottonWoolSpots",
            ):
                archive.writestr(f"train/{directory}/{key}.png", _png(mask))
        archive.writestr("test/Microaneurysms/heldout_sentinel.png", b"must not open")
    rows = ["name,x0,y0,x1,y1,H,W"]
    rows.extend(f"train_{index},2,2,30,30,32,32" for index in range(4))
    with zipfile.ZipFile(additional_path, "w") as archive:
        archive.writestr("MESSIDOR-ROIs.csv", "\n".join(rows) + "\n")


def _save_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, columns = np.ogrid[:64, :64]
    retina = (rows - 32) ** 2 + (columns - 32) ** 2 <= 27**2
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[retina] = np.array([100, 30, 20], dtype=np.uint8)
    Image.fromarray(image).save(path)


def _save_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[30:32, 30:32] = 255
    Image.fromarray(mask).save(path)


def _ddr(root: Path, *, confirmation_valid: bool = False) -> None:
    split_keys = {"train": ("train_a", "train_b"), "valid": ("valid_a",), "test": ("test_a",)}
    for split, keys in split_keys.items():
        grading = root / "DR_grading"
        (grading / split).mkdir(parents=True, exist_ok=True)
        (grading / f"{split}.txt").write_text(
            "".join(f"{key}.jpg 1\n" for key in keys), encoding="utf-8"
        )
        label_name = "segmentation label" if split == "valid" else "label"
        for key in keys:
            (grading / split / f"{key}.jpg").write_bytes(b"structure only")
            image_path = root / "lesion_segmentation" / split / "image" / f"{key}.png"
            if split == "train" or (split == "valid" and confirmation_valid):
                _save_rgb(image_path)
            else:
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(b"heldout sentinel")
            for axis in ("MA", "HE", "EX", "SE"):
                mask_path = (
                    root
                    / "lesion_segmentation"
                    / split
                    / label_name
                    / axis
                    / f"{key}.png"
                )
                if split == "train" or (split == "valid" and confirmation_valid):
                    _save_mask(mask_path)
                else:
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    mask_path.write_bytes(b"heldout sentinel")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preflight_then_run_never_opens_confirmation_or_test(tmp_path: Path) -> None:
    main_archive = tmp_path / "maples.zip"
    additional_archive = tmp_path / "additional.zip"
    _maples(main_archive, additional_archive)
    ddr_root = tmp_path / "ddr"
    _ddr(ddr_root)
    private_manifest = tmp_path / "manifest.csv"
    private_manifest.write_text("private manifest identity\n", encoding="utf-8")
    scales = [2.0 ** (-2.0 + index * 0.125) for index in range(33)]
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "development_boundary": {"maples_development_count": 2},
                "primary_scale_family": {
                    "policy": "MARGIN_ONLY",
                    "fine_scales": scales,
                    "coarse_stride": 2,
                },
                "reference_thresholds": {
                    "delta": {axis: 1.0 for axis in ("ma", "hem", "ex", "cws")},
                    "epsilon": {axis: 0.1 for axis in ("ma", "hem", "ex", "cws")},
                    "max_width": {axis: 0.0 for axis in ("ma", "hem", "ex", "cws")},
                    "numeric_tolerance": 1.0e-12,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    preflight = tmp_path / "preflight.json"
    common = [
        "--config",
        str(config),
        "--maples-archive",
        str(main_archive),
        "--maples-additional",
        str(additional_archive),
        "--ddr-root",
        str(ddr_root),
        "--ddr-private-manifest",
        str(private_manifest),
    ]
    assert main(["--mode", "preflight", *common, "--output", str(preflight)]) == 0
    preflight_payload = json.loads(preflight.read_text(encoding="utf-8"))
    assert preflight_payload["scientific_images_or_masks_opened"] is False
    lock = tmp_path / "lock.yaml"
    lock.write_text(
        yaml.safe_dump(
            {
                "lock_id": "synthetic",
                "input_identity": {
                    "maples_main_archive": _sha256(main_archive),
                    "maples_additional_archive": _sha256(additional_archive),
                    "ddr_private_manifest": _sha256(private_manifest),
                },
                "development_boundary": preflight_payload["development_boundary"],
                "held_out_or_test_scientific_access": "FORBIDDEN",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "result.json"
    assert main(
        [
            "--mode",
            "run",
            *common,
            "--input-lock",
            str(lock),
            "--output",
            str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "RPS_MED_R1_2_DEVELOPMENT_COMPLETE"
    assert payload["held_out_or_test_scientific_access"] is False
    assert payload["formal_r1"] == "NOT_RUN"
    assert "heldout_sentinel" not in json.dumps(payload)
    assert str(tmp_path) not in json.dumps(payload)


def test_confirmation_loaders_open_only_precommitted_scopes(tmp_path: Path) -> None:
    main_archive = tmp_path / "maples.zip"
    additional_archive = tmp_path / "additional.zip"
    _maples(main_archive, additional_archive)
    ddr_root = tmp_path / "ddr"
    _ddr(ddr_root, confirmation_valid=True)

    maples = load_maples_confirmation_measurements(
        main_archive, additional_archive, development_count=2
    )
    ddr = load_ddr_confirmation_measurements(ddr_root)

    assert maples.expected_input_count == 2
    assert len(maples.records) == 2
    assert maples.measurement_exclusion_count == 0
    assert ddr.expected_input_count == 1
    assert len(ddr.records) == 1
    assert ddr.measurement_exclusion_count == 0


def test_ddr_confirmation_manifest_validation_is_subset_scoped(tmp_path: Path) -> None:
    ddr_root = tmp_path / "ddr"
    _ddr(ddr_root, confirmation_valid=True)
    valid_files = tuple(
        path
        for path in (ddr_root / "lesion_segmentation" / "valid").rglob("*")
        if path.is_file()
    )
    manifest = tmp_path / "manifest.csv"
    rows = ["relative_path,size_bytes,sha256,status"]
    rows.extend(
        ",".join(
            (
                path.relative_to(ddr_root).as_posix(),
                str(path.stat().st_size),
                hashlib.sha256(path.read_bytes()).hexdigest(),
                "OK",
            )
        )
        for path in valid_files
    )
    rows.append("lesion_segmentation/test/image/test_a.png,1," + "0" * 64 + ",OK")
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")

    result = validate_ddr_confirmation_manifest(ddr_root, manifest)

    assert result["valid"] is True
    assert result["expected_file_count"] == len(valid_files)
    assert result["observed_file_count"] == len(valid_files)

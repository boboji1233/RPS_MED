"""Keep historical evidence checks optional in source-only distributions."""

from pathlib import Path

import pytest


EVIDENCE_CHECKS = {
    "test_v06_overlay_inherits_v05_science_and_rejects_extra_fields": (
        "metadata/gates/rps_med_r1_v05_formal_result_2026-08-25_v1.yaml",
    ),
    "test_v06_synthetic_diagnostics_are_deterministic_and_private": (
        "metadata/gates/rps_med_r1_v05_formal_result_2026-08-25_v1.yaml",
    ),
    "test_v07_config_discloses_only_fov_measurement_change": (
        "metadata/gates/rps_med_r1_v06_formal_result_2026-08-25_v1.yaml",
    ),
    "test_v07_synthetic_output_is_byte_deterministic": (
        "metadata/gates/rps_med_r1_v06_formal_result_2026-08-25_v1.yaml",
    ),
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    root = Path(config.rootpath)
    for item in items:
        required = EVIDENCE_CHECKS.get(item.name, ())
        if any(not (root / relative).is_file() for relative in required):
            item.add_marker(
                pytest.mark.skip(reason="Requires separately held historical execution evidence")
            )

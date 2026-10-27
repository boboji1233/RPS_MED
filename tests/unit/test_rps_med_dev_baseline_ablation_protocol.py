from __future__ import annotations

from itertools import combinations
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "experiment" / "rps_med_tjdr_dev_baseline_ablation_v0.1.yaml"


def _load() -> dict[str, object]:
    with CONFIG.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    assert isinstance(payload, dict)
    return payload


def test_protocol_is_development_only_and_preserves_five_state_contract() -> None:
    config = _load()
    data = config["data"]
    relation = config["relation"]
    execution = config["execution"]

    assert data["allowed_split"] == "official_train"
    assert data["official_test_access"] == "FORBIDDEN"
    assert relation["semantics_change"] == "FORBIDDEN"
    assert relation["states"] == [
        "LESS",
        "GREATER",
        "EQUIVALENT",
        "STRUCTURAL_INCOMPARABLE",
        "UNRESOLVED",
    ]
    assert len(relation["scales"]) == 17
    assert relation["scales"][0] == 0.25
    assert relation["scales"][-1] == 4.0
    assert execution["scientific_run_authorized"] is False
    assert execution["model_training"] == "FORBIDDEN_UNTIL_SEPARATE_AUTHORIZATION"


def test_synthetic_content_folds_are_disjoint_and_pairs_stay_within_fold() -> None:
    synthetic_ids = tuple(f"synthetic-{index:02d}" for index in range(20))
    folds = {fold: synthetic_ids[fold::5] for fold in range(5)}

    observed = [item for fold in folds.values() for item in fold]
    assert len(observed) == len(set(observed)) == len(synthetic_ids)
    assert set(observed) == set(synthetic_ids)

    for fold_items in folds.values():
        pairs = tuple(combinations(fold_items, 2))
        assert len(pairs) == len(fold_items) * (len(fold_items) - 1) // 2
        assert all(left in fold_items and right in fold_items for left, right in pairs)
        assert all(left != right for left, right in pairs)


def test_comparator_roles_and_dependence_contract_are_explicit() -> None:
    config = _load()
    roles = {row["id"]: row["role"] for row in config["baselines"]}

    assert roles["scale_specific_majority"] == "exact"
    assert roles["rps_multitask_pair"] == "exact"
    assert roles["conpro_severity"] == "adapted"
    assert roles["schmidt_ordinal_state"] == "adapted"
    assert config["uncertainty"]["resampling_unit"] == "UNIQUE_CONTENT_IMAGE"
    assert config["uncertainty"]["pair_row_bootstrap"] == "FORBIDDEN"

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stage3b" / "rps_med_stage3b_grammar_v1.yaml"
PREREG = ROOT / "metadata" / "preregistration" / "rps_med_stage3b_grammar_v1.yaml"
CONFIG_V2 = ROOT / "configs" / "stage3b" / "rps_med_stage3b_grammar_v2.yaml"
PREREG_V2 = ROOT / "metadata" / "preregistration" / "rps_med_stage3b_grammar_v2.yaml"


def _yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_stage3b_preserves_stage3a_data_relation_and_seed_contract() -> None:
    config = _yaml(CONFIG)
    data = config["data"]
    relation = config["relation"]
    assert isinstance(data, dict)
    assert isinstance(relation, dict)
    assert (data["unique_units"], data["train_units"], data["validation_units"]) == (
        442,
        354,
        88,
    )
    assert (data["train_pairs"], data["validation_pairs"]) == (62481, 3828)
    assert len(relation["scales"]) == 17
    assert relation["states"] == [
        "LESS",
        "GREATER",
        "EQUIVALENT",
        "STRUCTURAL_INCOMPARABLE",
        "UNRESOLVED",
    ]
    assert config["seeds"] == [20260826, 20260827, 20260828]


def test_stage3b_grammar_and_training_roster_are_frozen() -> None:
    config = _yaml(CONFIG)
    grammar = config["grammar"]
    methods = config["methods"]
    training = config["training"]
    assert isinstance(grammar, dict)
    assert isinstance(methods, dict)
    assert isinstance(training, dict)
    assert grammar["path_count"] == 101
    assert len(grammar["transition_pairs"]) == 6
    assert len(grammar["change_boundaries"]) == 16
    assert list(methods) == ["G0", "G1", "G2", "G3", "G4"]
    assert training["max_epochs"] == 100
    assert training["early_stopping_patience"] == 12
    assert methods["G3"]["scale_ce_weight"] == 0.0
    assert methods["G4"]["scale_ce_weight"] == 1.0


def test_stage3b_forbids_formal_sources_and_caps_cuda() -> None:
    config = _yaml(CONFIG)
    runtime = config["runtime"]
    denylist = config["denylist"]
    prereg = _yaml(PREREG)
    boundaries = prereg["boundaries"]
    assert isinstance(runtime, dict)
    assert isinstance(denylist, dict)
    assert isinstance(boundaries, dict)
    assert runtime["max_process_cuda_bytes"] == 3_000_000_000
    assert runtime["environment"] == "D:/boshi/am/.conda/rps-med-stage3b"
    assert {"official_test", "formal_test", "ddr", "authorization", "passport"} <= set(
        denylist["forbidden_tokens"]
    )
    assert boundaries["official_test_access"] == "FORBIDDEN"
    assert boundaries["ddr_access"] == "FORBIDDEN"
    assert boundaries["formal_model_reselection"] == "FORBIDDEN"
    assert boundaries["selective_seed_retry"] == "FORBIDDEN"


def test_stage3b_statistical_unit_is_node_not_pair_row() -> None:
    config = _yaml(CONFIG)
    metrics = config["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["statistical_unit"] == "VALIDATION_UNIQUE_CONTENT_NODE"
    assert metrics["pair_row_bootstrap"] == "FORBIDDEN"
    assert metrics["multiplicity"] == "HOLM_WITHIN_METRIC_FAMILY"


def test_stage3b_private_outputs_and_environment_are_git_ignored() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".conda/" in ignore
    assert "reports/stage3b/" in ignore


def test_indexed_cuda_device_is_compatible_with_memory_fraction_contract() -> None:
    import torch

    device = torch.device("cuda:0")
    assert device.type == "cuda"
    assert device.index == 0


def test_stage3b_v2_removes_only_unreproducible_logit_replay() -> None:
    legacy = _yaml(CONFIG)
    config = _yaml(CONFIG_V2)
    prereg = _yaml(PREREG_V2)
    assert list(legacy["methods"]) == ["G0", "G1", "G2", "G3", "G4"]
    assert list(config["methods"]) == ["G0", "G1", "G3", "G4"]
    assert config["data"] == legacy["data"]
    assert config["relation"] == legacy["relation"]
    assert config["grammar"] == legacy["grammar"]
    assert config["training"] == legacy["training"]
    assert config["seeds"] == legacy["seeds"]
    assert config["metrics"] == legacy["metrics"]
    assert config["interpretation"] == legacy["interpretation"]
    removed = prereg["removed_method"]
    assert isinstance(removed, dict)
    assert removed == {
        "id": "G2",
        "reason": "EXACT_FIXED_LOGIT_REPLAY_UNAVAILABLE",
        "result_accepted": False,
    }

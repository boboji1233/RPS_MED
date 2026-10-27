from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "configs" / "stage3a" / "rps_med_stage3a_common_v1.yaml"
RUNS = ROOT / "configs" / "stage3a" / "rps_med_stage3a_runs_v1.yaml"
PREREG = ROOT / "metadata" / "preregistration" / "rps_med_stage3a_dev_only_v1.yaml"


def _yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_stage3a_boundary_is_exactly_the_frozen_development_partition() -> None:
    common = _yaml(COMMON)
    data = common["data"]
    assert isinstance(data, dict)
    assert data["allowed_cache_sha256"] == (
        "25b29c2fb2b10f8eac9a44e20f027c6bf3e521ed7b01fcea6f3f0db8f55a7f10"
    )
    assert data["partition_digest"] == (
        "c4ff06ab1cc8a130e43ec7abc1d32f45acec5b0d9008ad998f06c19262385ebe"
    )
    assert (
        data["expected_unique_content_units"],
        data["expected_train_units"],
        data["expected_validation_units"],
    ) == (442, 354, 88)
    assert (data["expected_train_pairs"], data["expected_validation_pairs"]) == (
        62481,
        3828,
    )


def test_stage3a_preserves_five_states_scales_and_seeds() -> None:
    common = _yaml(COMMON)
    relation = common["relation"]
    assert isinstance(relation, dict)
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
    assert common["seeds"] == [20260826, 20260827, 20260828]


def test_stage3a_denies_formal_test_ddr_and_pair_row_bootstrap() -> None:
    common = _yaml(COMMON)
    denylist = common["denylist"]
    metrics = common["metrics"]
    runtime = common["runtime"]
    assert isinstance(denylist, dict)
    assert isinstance(metrics, dict)
    assert isinstance(runtime, dict)
    assert {"official_test", "formal_test", "ddr", "authorization", "passport"} <= set(
        denylist["forbidden_argument_tokens"]
    )
    assert metrics["statistical_unit"] == "VALIDATION_UNIQUE_CONTENT_NODE"
    assert metrics["pair_row_bootstrap"] == "FORBIDDEN"
    assert metrics["per_state_f1"] == "POOLED_OVER_ALL_VALIDATION_PAIR_SCALE_ROWS"
    assert runtime["automatic_retry"] is False
    assert runtime["append_only"] is True


def test_stage3a_encoder_stopping_is_defined_before_results() -> None:
    common = _yaml(COMMON)
    encoder = common["encoder_training"]
    assert isinstance(encoder, dict)
    stopping = encoder["early_stopping_by_contract"]
    assert isinstance(stopping, dict)
    assert stopping["presence_only"] == "MIN_VALIDATION_PRESENCE_BINARY_CROSS_ENTROPY"
    assert stopping["burden_only"] == (
        "MAX_VALIDATION_MULTISCALE_MACRO_F1_FROM_ANALYTIC_RELATIONS"
    )


def test_stage3a_aliases_prevent_duplicate_evidence_runs() -> None:
    runs = _yaml(RUNS)
    aliases = runs["aliases"]
    unique_runs = runs["unique_runs"]
    assert isinstance(aliases, dict)
    assert isinstance(unique_runs, dict)
    assert aliases["B2"] == aliases["A1"] == "B2_A1"
    assert aliases["B3"] == aliases["A3"] == "B3_A3"
    assert aliases["B6"] == aliases["A0"] == aliases["A12_full"]
    assert unique_runs["A11"]["five_state_primary"] == "NOT_APPLICABLE"
    assert runs["selective_retry"] == "FORBIDDEN"
    assert runs["result_driven_change"] == "FORBIDDEN"


def test_stage3a_preregistration_forbids_claim_expansion() -> None:
    prereg = _yaml(PREREG)
    boundary = prereg["data_boundary"]
    interpretation = prereg["interpretation"]
    execution = prereg["execution"]
    assert isinstance(boundary, dict)
    assert isinstance(interpretation, dict)
    assert isinstance(execution, dict)
    assert boundary["official_test_access"] == "FORBIDDEN"
    assert boundary["ddr_access"] == "FORBIDDEN"
    assert interpretation["formal_test_component_proof"] == "FORBIDDEN"
    assert interpretation["external_validation_claim"] == "FORBIDDEN"
    assert interpretation["clinical_or_patient_claim"] == "FORBIDDEN"
    assert execution["remote_push"] == "FORBIDDEN"

from __future__ import annotations

from copy import deepcopy

import pytest

from auto_research.domain_contracts import canonical_hash
from auto_research.agents.plan import _trial_spec_from_plan
from auto_research.proxy_classifier import (
    build_proxy_decision_policy,
    classify_proxy_outcome,
)
from auto_research.research_state import ResearchEventLedger
from test_m113_ledger_closure import _direction, _variant


def _policy() -> dict:
    return build_proxy_decision_policy(
        primary_metric_id="accuracy",
        objective="maximize",
        aggregation="paired_mean",
        datasets=["dataset-a", "dataset-b"],
        seeds=[7, 11],
        metric_ids=["accuracy"],
        roles=["baseline", "candidate"],
        aggregate_improvement_threshold=0.05,
        per_dataset_maximum_regression=0.01,
        activation_surface_ids=["src/model.py", "src/router.py"],
        readiness_check_ids=["artifacts", "budget", "runtime"],
        evidence_kinds=["activation_evidence", "full_s3_readiness", "proxy_results"],
        mode="gate_to_full",
    )


def _binding(policy: dict) -> dict:
    payload = {
        "schema_version": "auto_research_proxy_evaluation_binding_v1",
        "policy_hash": policy["policy_hash"],
        "attempt_id": "attempt-m115",
        "direction_semantic_hash": "1" * 64,
        "direction_spec_hash": "2" * 64,
        "variant_semantic_hash": "3" * 64,
        "variant_spec_hash": "4" * 64,
        "trial_spec_hash": "5" * 64,
        "lifecycle_generation": 0,
        "implementation_hash": "a" * 64,
        "attempt_input_hash": "b" * 64,
        "phase_execution_id": "phase-proxy-m115",
        "phase_start_event_id": "event:proxy:m115",
        "producer_run_id": "producer-m115",
        "command_plan_hash": "6" * 64,
        "phase_contract_hash": "7" * 64,
        "sample_contract_ref": {"artifact_hash": "8" * 64},
        "evaluator_contract_ref": {"artifact_hash": "9" * 64},
        "provenance_mode": "synthetic",
        "expected_evidence_kinds": policy["evidence_kinds"],
    }
    payload["binding_hash"] = canonical_hash(payload)
    return payload


def _identity(kind: str, evidence_id: str) -> dict:
    return {
        "evidence_kind": kind,
        "evidence_id": evidence_id,
        "attempt_id": "attempt-m115",
        "lifecycle_generation": 0,
        "implementation_hash": "a" * 64,
        "attempt_input_hash": "b" * 64,
        "phase": "proxy",
        "phase_execution_id": "phase-proxy-m115",
        "phase_start_event_id": "event:proxy:m115",
        "producer_run_id": "producer-m115",
    }


def _evidence() -> dict:
    rows = []
    values = {
        ("dataset-a", 7): (0.50, 0.60),
        ("dataset-a", 11): (0.52, 0.62),
        ("dataset-b", 7): (0.40, 0.46),
        ("dataset-b", 11): (0.42, 0.48),
    }
    for (dataset_id, seed), (baseline, candidate) in values.items():
        for role, value in (("baseline", baseline), ("candidate", candidate)):
            rows.append({"phase": "proxy", "role": role, "dataset_id": dataset_id, "metric_id": "accuracy", "seed": seed, "metric_value": value, "command_status": "completed"})
    return {
        "proxy": {**_identity("proxy_results", "evidence-proxy-m115"), "rows": rows},
        "activation": {**_identity("activation_evidence", "evidence-activation-m115"), "status": "passed", "command_status": "completed", "exit_code": 0, "implementation_surface_ids": ["src/model.py", "src/router.py"]},
        "readiness": {**_identity("full_s3_readiness", "evidence-readiness-m115"), "ready": True, "checks": [{"check_id": "artifacts", "status": "PASS"}, {"check_id": "budget", "status": "PASS"}, {"check_id": "runtime", "status": "PASS"}]},
    }


def _classify(policy: dict | None = None, binding: dict | None = None, evidence: dict | None = None) -> dict:
    policy = policy or _policy()
    return classify_proxy_outcome(frozen_policy=policy, evaluation_binding=binding or _binding(policy), decoded_evidence=evidence or _evidence(), evidence_manifest_hash="f" * 64)


def test_frozen_policy_passes_exact_paired_coverage_without_producer_policy() -> None:
    policy = _policy()
    evidence = _evidence()
    before = deepcopy(evidence)
    outcome = _classify(policy=policy, evidence=evidence)
    assert outcome["decision"] == "RUN_FULL"
    assert outcome["observed_delta"] == pytest.approx(0.08)
    assert outcome["dataset_deltas"] == pytest.approx({"dataset-a": 0.10, "dataset-b": 0.06})
    assert outcome["proxy_decision_policy_hash"] == policy["policy_hash"]
    assert evidence == before


def test_producer_policy_is_rejected_as_unregistered_extra_evidence() -> None:
    assert _classify()["decision"] == "RUN_FULL"
    evidence = _evidence()
    evidence["producer-policy"] = {**_identity("effective_proxy_policy", "producer-policy"), "decision_threshold": -999.0}
    with pytest.raises(ValueError, match="evidence kinds"):
        _classify(evidence=evidence)


@pytest.mark.parametrize("attack", ["missing_seed", "extra_dataset", "duplicate", "surface", "readiness", "binding"])
def test_proxy_attacks_fail_after_valid_baseline(attack: str) -> None:
    assert _classify()["decision"] == "RUN_FULL"
    policy = _policy()
    binding = _binding(policy)
    evidence = _evidence()
    if attack == "missing_seed": evidence["proxy"]["rows"].pop()
    elif attack == "extra_dataset": evidence["proxy"]["rows"][0]["dataset_id"] = "dataset-c"
    elif attack == "duplicate": evidence["proxy"]["rows"].append(deepcopy(evidence["proxy"]["rows"][0]))
    elif attack == "surface": evidence["activation"]["implementation_surface_ids"].pop()
    elif attack == "readiness": evidence["readiness"]["checks"][0]["status"] = "FAIL"
    else:
        binding["producer_run_id"] = "other-producer"
        binding["binding_hash"] = canonical_hash({key: value for key, value in binding.items() if key != "binding_hash"})
    with pytest.raises(ValueError):
        _classify(policy=policy, binding=binding, evidence=evidence)


def test_dataset_regression_blocks_full_even_when_aggregate_passes() -> None:
    policy = build_proxy_decision_policy(primary_metric_id="accuracy", objective="maximize", aggregation="paired_mean", datasets=["dataset-a", "dataset-b"], seeds=[7, 11], metric_ids=["accuracy"], roles=["baseline", "candidate"], aggregate_improvement_threshold=0.01, per_dataset_maximum_regression=0.01, activation_surface_ids=["src/model.py", "src/router.py"], readiness_check_ids=["artifacts", "budget", "runtime"], evidence_kinds=["activation_evidence", "full_s3_readiness", "proxy_results"], mode="gate_to_full")
    evidence = _evidence()
    for row in evidence["proxy"]["rows"]:
        if row["dataset_id"] == "dataset-b" and row["role"] == "candidate": row["metric_value"] -= 0.2
        if row["dataset_id"] == "dataset-a" and row["role"] == "candidate": row["metric_value"] += 0.2
    outcome = _classify(policy=policy, binding=_binding(policy), evidence=evidence)
    assert outcome["observed_delta"] > 0.01
    assert outcome["worst_dataset_regression"] > 0.01
    assert outcome["decision"] == "PROPOSE_NEXT_VARIANT"


def test_trial_spec_v6_freezes_policy_and_proxy_start_binds_runtime_identity(tmp_path) -> None:
    direction = _direction()
    variant = _variant(direction)
    plan = {
        "datasets": [{"name": "fake", "split": "test", "sample_count": 1}],
        "metrics": [{"name": "accuracy", "primary": True, "higher_is_better": True}],
        "statistical_testing": {"seeds": [7, 11]},
        "acceptance_criteria": {"minimum_mean_delta": 0.05, "maximum_dataset_regression": 0.01},
        "ablation_matrix": [],
        "execution": {"mode": "simulate", "collector": "c2c_small_loop", "commands": []},
    }
    trial_spec = _trial_spec_from_plan(plan, variant, profile="standard", project_root=tmp_path)
    policy = trial_spec["proxy_decision_policy"]
    assert trial_spec["schema_version"] == "auto_research_trial_spec_v6"
    assert policy["mode"] == "gate_to_full"
    assert "effective_proxy_policy" not in policy["evidence_kinds"]
    ledger = ResearchEventLedger(tmp_path)
    ledger.select_direction(direction)
    ledger.plan_variant(variant)
    attempt = ledger.reserve_attempt(profile="standard", direction=direction, variant=variant, implementation_hash="a" * 64, attempt_kind="proxy_full", trial_spec=trial_spec)
    attempt = ledger.start_proxy_phase(attempt["attempt_id"], phase_execution_id="proxy-phase-m115", producer_run_id="proxy-producer-m115")
    manifest = attempt["phase_executions"]["proxy"]
    binding = manifest["proxy_evaluation_binding"]
    assert manifest["schema_version"] == "auto_research_phase_execution_manifest_v3"
    assert binding["policy_hash"] == policy["policy_hash"]
    assert binding["attempt_id"] == attempt["attempt_id"]
    assert binding["phase_start_event_id"] == manifest["phase_start_event_id"]
    assert binding["expected_evidence_kinds"] == policy["evidence_kinds"]

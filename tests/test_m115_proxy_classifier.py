from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType

import pytest

from auto_research.domain_contracts import canonical_hash
from auto_research.proxy_classifier import classify_proxy_outcome


def _contract() -> dict:
    payload = {
        "schema_version": "auto_research_proxy_decision_contract_v1",
        "attempt_id": "attempt-m115",
        "lifecycle_generation": 0,
        "implementation_hash": "a" * 64,
        "attempt_input_hash": "b" * 64,
        "phase_execution_id": "phase-proxy-m115",
        "phase_start_event_id": "event:proxy:m115",
        "primary_metric_id": "accuracy",
        "metric_ids": ["accuracy"],
        "objective": "maximize",
        "datasets": ["dataset-a", "dataset-b"],
        "seeds": [7, 11],
        "roles": ["baseline", "candidate"],
        "evidence_kinds": ["activation_evidence", "effective_proxy_policy", "full_s3_readiness", "proxy_results"],
        "activation_surface_ids": ["src/model.py", "src/router.py"],
        "readiness_check_ids": ["artifacts", "budget", "runtime"],
        "constraints": [
            {
                "constraint_id": "proxy-mean",
                "kind": "minimum_mean_delta",
                "hard": True,
                "metric_id": "accuracy",
                "threshold": 0.05,
                "objective": "maximize",
            },
            {
                "constraint_id": "proxy-regression",
                "kind": "per_dataset_maximum_regression",
                "hard": True,
                "metric_id": "accuracy",
                "threshold": 0.01,
                "objective": "maximize",
            },
        ],
    }
    payload["contract_hash"] = canonical_hash(payload)
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
            rows.append(
                {
                    "phase": "proxy",
                    "role": role,
                    "dataset_id": dataset_id,
                    "metric_id": "accuracy",
                    "seed": seed,
                    "metric_value": value,
                    "command_status": "completed",
                }
            )
    return {
        "evidence-proxy-m115": {**_identity("proxy_results", "evidence-proxy-m115"), "rows": rows},
        "evidence-activation-m115": {
            **_identity("activation_evidence", "evidence-activation-m115"),
            "status": "passed",
            "command_status": "completed",
            "exit_code": 0,
            "implementation_surface_ids": ["src/model.py", "src/router.py"],
        },
        "evidence-readiness-m115": {
            **_identity("full_s3_readiness", "evidence-readiness-m115"),
            "ready": True,
            "checks": [
                {"check_id": "artifacts", "status": "PASS"},
                {"check_id": "budget", "status": "PASS"},
                {"check_id": "runtime", "status": "PASS"},
            ],
        },
        "evidence-policy-m115": {
            **_identity("effective_proxy_policy", "evidence-policy-m115"),
            "decision_threshold": 999.0,
        },
    }


def _classify(contract: dict | None = None, evidence: dict | None = None) -> dict:
    return classify_proxy_outcome(
        frozen_contract=_freeze(contract or _contract()),
        decoded_evidence=_freeze(evidence or _evidence()),
    )


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def test_baseline_passes_with_exact_paired_coverage_and_ignores_producer_threshold() -> None:
    contract = _contract()
    evidence = _evidence()
    before_contract = deepcopy(contract)
    before_evidence = deepcopy(evidence)

    outcome = _classify(contract, evidence)

    assert outcome["decision"] == "RUN_FULL"
    assert outcome["observed_delta"] == pytest.approx(0.08)
    assert outcome["dataset_deltas"] == pytest.approx({"dataset-a": 0.10, "dataset-b": 0.06})
    assert outcome["worst_dataset_regression"] == 0.0
    assert [item["status"] for item in outcome["constraint_results"]] == ["PASS", "PASS"]
    assert contract == before_contract
    assert evidence == before_evidence


@pytest.mark.parametrize(
    ("attack", "message"),
    [
        ("missing_kind", "evidence kinds"),
        ("extra_kind", "evidence kinds"),
        ("duplicate_kind", "duplicate decoded evidence kind"),
        ("missing_pair", "row coverage"),
        ("extra_dataset", "row coverage"),
        ("wrong_phase", "proxy-phase evidence"),
        ("activation_surface", "activation surfaces"),
        ("readiness_check", "readiness checks"),
        ("readiness_status", "did not pass"),
    ],
)
def test_single_point_attacks_fail_after_a_passing_baseline(attack: str, message: str) -> None:
    assert _classify()["decision"] == "RUN_FULL"
    evidence = _evidence()
    if attack == "missing_kind":
        del evidence["evidence-policy-m115"]
    elif attack == "extra_kind":
        evidence["evidence-extra-m115"] = _identity("proxy_cache_report", "evidence-extra-m115")
    elif attack == "duplicate_kind":
        evidence["evidence-proxy-copy"] = {**evidence["evidence-proxy-m115"], "evidence_id": "evidence-proxy-copy"}
    elif attack == "missing_pair":
        evidence["evidence-proxy-m115"]["rows"].pop()
    elif attack == "extra_dataset":
        evidence["evidence-proxy-m115"]["rows"][0]["dataset_id"] = "dataset-c"
    elif attack == "wrong_phase":
        evidence["evidence-policy-m115"]["phase"] = "full"
    elif attack == "activation_surface":
        evidence["evidence-activation-m115"]["implementation_surface_ids"] = ["src/model.py"]
    elif attack == "readiness_check":
        evidence["evidence-readiness-m115"]["checks"].pop()
    else:
        evidence["evidence-readiness-m115"]["checks"][0]["status"] = "FAIL"

    with pytest.raises(ValueError, match=message):
        _classify(evidence=evidence)


def test_dataset_regression_uses_paired_objective_normalized_deltas() -> None:
    assert _classify()["decision"] == "RUN_FULL"
    evidence = _evidence()
    for row in evidence["evidence-proxy-m115"]["rows"]:
        if row["dataset_id"] == "dataset-b" and row["role"] == "candidate":
            row["metric_value"] -= 0.08

    outcome = _classify(evidence=evidence)

    assert outcome["observed_delta"] == pytest.approx(0.04)
    assert outcome["dataset_deltas"] == pytest.approx({"dataset-a": 0.10, "dataset-b": -0.02})
    assert outcome["worst_dataset_regression"] == pytest.approx(0.02)
    assert outcome["decision"] == "PROPOSE_NEXT_VARIANT"
    assert [item["status"] for item in outcome["constraint_results"]] == ["FAIL", "FAIL"]

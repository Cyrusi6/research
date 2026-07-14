from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from auto_research.agents.experiment import (
    _identity_evidence_payload,
    _quantitative_evidence_payload,
    _stage_evidence_inventory,
    _write_staged_evidence_source,
)
from auto_research.domain_contracts import canonical_hash
from auto_research.evidence import EVIDENCE_SCHEMA_VERSIONS, encode_canonical_evidence
from auto_research.research_state import IntegrityError, ResearchEventLedger
from auto_research.utils import write_json
from auto_research.validators import run_stage_gate
from test_authoritative_state_machine import _attempt_inputs, _direction, _initialize, _reserve, _trial_spec, _variant


_BOOTSTRAP_EVIDENCE_KINDS = (
    "main_results",
    "activation_evidence",
    "proxy_baseline_fingerprint",
    "proxy_cache_report",
    "effective_proxy_policy",
    "proxy_calibration_policy",
    "proxy_decision_report",
    "bootstrap_completion",
)


def _bootstrap_trial_spec() -> dict:
    trial_spec = _trial_spec(profile="bootstrap", attempt_kind="bootstrap_proxy")
    trial_spec["required_artifacts"] = list(_BOOTSTRAP_EVIDENCE_KINDS)
    trial_spec["evidence_requirements"] = [
        {
            "requirement_id": kind.replace("_", "-"),
            "kind": kind,
            "required": True,
            "applicable_phases": ["proxy"] if kind != "activation_evidence" else ["always"],
            "schema_version": EVIDENCE_SCHEMA_VERSIONS[kind],
        }
        for kind in _BOOTSTRAP_EVIDENCE_KINDS
    ]
    return trial_spec


def _reserve_bootstrap(ledger: ResearchEventLedger, direction: dict, variant: dict) -> tuple[dict, dict]:
    trial_spec = _bootstrap_trial_spec()
    write_json(ledger.project_root / "plan" / "trial_spec.json", trial_spec)
    attempt = ledger.reserve_attempt(
        profile="bootstrap",
        direction=direction,
        variant=variant,
        attempt_kind="bootstrap_proxy",
        trial_spec=trial_spec,
        **_attempt_inputs(variant),
    )
    return attempt, trial_spec


def _payload_hash(payload: dict) -> str:
    return hashlib.sha256(encode_canonical_evidence(payload)).hexdigest()


def _bootstrap_inventory(root: Path, attempt: dict, trial_spec: dict) -> list[dict]:
    producer_run_id = f"bootstrap-run-{attempt['attempt_id'][:12]}"
    payloads: list[tuple[str, dict]] = []

    main = _quantitative_evidence_payload(
        attempt=attempt,
        trial_spec=trial_spec,
        producer_run_id=producer_run_id,
        evidence_kind="main_results",
        phase="proxy",
        role_values={"baseline": 0.0, "candidate": 1.0},
    )
    payloads.append(("main_results", main))
    main_hash = _payload_hash(main)

    activation = _identity_evidence_payload(
        attempt=attempt,
        producer_run_id=producer_run_id,
        evidence_kind="activation_evidence",
        fields={
            "probe_id": "bootstrap-activation-probe",
            "status": "passed",
            "command_status": "completed",
            "exit_code": 0,
            "implementation_surface_ids": ["src/router.py"],
        },
    )
    payloads.append(("activation_evidence", activation))

    fingerprint_inputs = {
        "sample_manifest_hash": attempt["sample_manifest_hash"],
        "evaluator_hash": attempt["evaluator_hash"],
        "protocol_hash": attempt["protocol_hash"],
    }
    baseline = _identity_evidence_payload(
        attempt=attempt,
        producer_run_id=producer_run_id,
        evidence_kind="proxy_baseline_fingerprint",
        fields={
            "baseline_hash": canonical_hash(fingerprint_inputs),
            "dataset_ids": [item["dataset_id"] for item in trial_spec["datasets"]],
            "seeds": list(trial_spec["statistical_testing"]["seeds"]),
            "fingerprint_inputs": fingerprint_inputs,
        },
    )
    payloads.append(("proxy_baseline_fingerprint", baseline))
    baseline_hash = _payload_hash(baseline)

    cache = _identity_evidence_payload(
        attempt=attempt,
        producer_run_id=producer_run_id,
        evidence_kind="proxy_cache_report",
        fields={
            "cross_references": {"proxy_baseline_fingerprint_hash": baseline_hash},
            "cache_key": canonical_hash({"baseline": baseline_hash, "attempt": attempt["attempt_id"]}),
            "baseline_hash": baseline["baseline_hash"],
            "cache_entry_hash": canonical_hash({"cache": "bootstrap", "baseline": baseline_hash}),
            "status": "hit",
        },
    )
    payloads.append(("proxy_cache_report", cache))
    cache_hash = _payload_hash(cache)

    policy_body = {
        "required_phases": ["proxy"],
        "proxy_terminal_allowed": True,
        "decision_threshold": 0.05,
    }
    policy = _identity_evidence_payload(
        attempt=attempt,
        producer_run_id=producer_run_id,
        evidence_kind="effective_proxy_policy",
        fields={"policy_hash": canonical_hash(policy_body), **policy_body},
    )
    payloads.append(("effective_proxy_policy", policy))
    policy_hash = _payload_hash(policy)

    calibration_refs = {
        "proxy_baseline_fingerprint_hash": baseline_hash,
        "effective_proxy_policy_hash": policy_hash,
    }
    calibration_body = {
        "status": "calibrated",
        "calibration_metric": trial_spec["primary_metric_id"],
        "calibration_value": 1.0,
        "cross_references": calibration_refs,
    }
    calibration = _identity_evidence_payload(
        attempt=attempt,
        producer_run_id=producer_run_id,
        evidence_kind="proxy_calibration_policy",
        fields={"calibration_hash": canonical_hash(calibration_body), **calibration_body},
    )
    payloads.append(("proxy_calibration_policy", calibration))
    calibration_hash = _payload_hash(calibration)

    decision_refs = {
        "proxy_baseline_fingerprint_hash": baseline_hash,
        "proxy_cache_report_hash": cache_hash,
        "effective_proxy_policy_hash": policy_hash,
        "proxy_calibration_policy_hash": calibration_hash,
        "main_results_hash": main_hash,
    }
    decision = _identity_evidence_payload(
        attempt=attempt,
        producer_run_id=producer_run_id,
        evidence_kind="proxy_decision_report",
        fields={
            "cross_references": decision_refs,
            "decision": "terminal_proxy",
            "reason_codes": ["bootstrap_proxy_verified"],
            "observed_proxy_delta": 1.0,
        },
    )
    payloads.append(("proxy_decision_report", decision))
    decision_hash = _payload_hash(decision)

    completion = _identity_evidence_payload(
        attempt=attempt,
        producer_run_id=producer_run_id,
        evidence_kind="bootstrap_completion",
        fields={
            "cross_references": {
                "proxy_decision_report_hash": decision_hash,
                "main_results_hash": main_hash,
            },
            "completion_status": "verified",
            "phase": "proxy",
        },
    )
    payloads.append(("bootstrap_completion", completion))

    return [
        _write_staged_evidence_source(
            root,
            producer_run_id=producer_run_id,
            evidence_kind=kind,
            payload=payload,
        )
        for kind, payload in payloads
    ]


def _prepare_s3_outputs(root: Path, *, bootstrap: bool = False) -> tuple[ResearchEventLedger, dict, dict, dict, dict]:
    direction = _direction()
    variant = _variant(direction, 1)
    write_json(root / "literature" / "direction.json", direction)
    write_json(root / "plan" / "variant.json", variant)
    ledger = ResearchEventLedger(root)
    _initialize(ledger, direction, variant)
    if bootstrap:
        attempt, trial_spec = _reserve_bootstrap(ledger, direction, variant)
        inventory = _bootstrap_inventory(root, attempt, trial_spec)
        completion = _stage_evidence_inventory(
            project_root=root,
            attempt=attempt,
            trial_spec=trial_spec,
            inventory=inventory,
        )
        ledger.transition_attempt(attempt["attempt_id"], "PROXY_RUNNING", phase="proxy", phase_state="RUNNING")
        ledger.complete_attempt(completion)
    else:
        attempt = _reserve(ledger, direction, variant)
        from test_authoritative_state_machine import _complete

        _complete(ledger, attempt, outcome="accepted")
        trial_spec = attempt["frozen_trial_spec"]
    (root / "experiment" / "results" / "hypothesis_verification.md").write_text("ok\n", encoding="utf-8")
    return ledger, direction, variant, attempt, trial_spec


def test_s3_gate_passes_strict_trial_result(tmp_path: Path) -> None:
    _prepare_s3_outputs(tmp_path)
    report = run_stage_gate("S3_experiment", tmp_path, {}).to_dict()
    assert report["status"] == "PASS"
    assert next(check for check in report["checks"] if check["name"] == "s3_authoritative_transaction")["status"] == "PASS"


def test_s3_gate_retries_without_trial_result(tmp_path: Path) -> None:
    direction = _direction()
    variant = _variant(direction, 1)
    write_json(tmp_path / "literature" / "direction.json", direction)
    write_json(tmp_path / "plan" / "variant.json", variant)
    report = run_stage_gate("S3_experiment", tmp_path, {}).to_dict()
    assert report["status"] == "NEEDS_RETRY"
    assert any(check.get("name") == "s3_authoritative_transaction" for check in report["checks"])


def test_s3_bootstrap_gate_is_repeatable_read_only_and_budget_isolated(tmp_path: Path) -> None:
    ledger, direction, _, attempt, _ = _prepare_s3_outputs(tmp_path, bootstrap=True)
    config = {"orchestration": {"profile": "bootstrap"}}
    before_events = ledger.events()
    before_state = ledger.state()

    first = run_stage_gate("S3_experiment", tmp_path, config).to_dict()
    second = run_stage_gate("S3_experiment", tmp_path, config).to_dict()
    after_events = ledger.events()
    state = ledger.state()

    assert first["status"] == second["status"] == "PASS"
    assert len(after_events) == len(before_events)
    assert state["last_sequence"] == before_state["last_sequence"]
    committed_attempt = state["attempts"][attempt["attempt_id"]]
    assert committed_attempt["state"] == "METHOD_COMPLETED"
    assert committed_attempt["profile"] == "bootstrap"
    assert committed_attempt["attempt_kind"] == "bootstrap_proxy"
    assert committed_attempt["consumes_direction_budget"] is False
    assert committed_attempt["reserved_slot"] is False
    assert state["directions"][direction["direction_semantic_hash"]]["budget"] == {"target": 5, "reserved": 0, "consumed": 0}
    assert not any(item.get("consumes_direction_budget") for item in state["method_tried_history"])
    assert state["latest_direction_aggregate"] is None
    assert state["directions"][direction["direction_semantic_hash"]]["status"] == "ACTIVE"
    assert list(state["trial_results"]) == [attempt["attempt_id"]]
    assert state["last_route_outcome"]["next_action"] == "FINISH_RUN"
    assert state["last_route_outcome"]["source"]["attempt_id"] == attempt["attempt_id"]
    assert sum(event["event_type"] == "AttemptFinalized" for event in after_events) == 1


@pytest.mark.parametrize("missing_kind", _BOOTSTRAP_EVIDENCE_KINDS)
def test_bootstrap_missing_preregistered_evidence_is_zero_write(tmp_path: Path, missing_kind: str) -> None:
    direction = _direction()
    variant = _variant(direction, 1)
    write_json(tmp_path / "literature" / "direction.json", direction)
    write_json(tmp_path / "plan" / "variant.json", variant)
    ledger = ResearchEventLedger(tmp_path)
    _initialize(ledger, direction, variant)
    attempt, trial_spec = _reserve_bootstrap(ledger, direction, variant)
    inventory = _bootstrap_inventory(tmp_path, attempt, trial_spec)
    completion = _stage_evidence_inventory(
        project_root=tmp_path,
        attempt=attempt,
        trial_spec=trial_spec,
        inventory=inventory,
    )
    ledger.transition_attempt(attempt["attempt_id"], "PROXY_RUNNING", phase="proxy", phase_state="RUNNING")
    invalid = deepcopy(completion)
    invalid["entries"] = [entry for entry in invalid["entries"] if entry["kind"] != missing_kind]
    before_events = ledger.events()
    before_state = ledger.state()

    with pytest.raises(IntegrityError):
        ledger.complete_attempt(invalid)

    after_state = ledger.state()
    assert ledger.events() == before_events
    assert after_state["last_sequence"] == before_state["last_sequence"]
    assert after_state["trial_results"] == {}
    assert after_state["last_route_outcome"] == before_state["last_route_outcome"]
    assert after_state["method_tried_history"] == []
    assert after_state["directions"][direction["direction_semantic_hash"]]["budget"] == {"target": 5, "reserved": 0, "consumed": 0}

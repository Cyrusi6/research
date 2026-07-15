from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from auto_research.agents.experiment import (
    _c2c_strict_evidence_inventory,
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
from support.authoritative_evidence import record_completed_evidence_command, start_attempt_phase


_BOOTSTRAP_EVIDENCE_KINDS = (
    "proxy_results",
    "activation_evidence",
    "proxy_baseline_fingerprint",
    "proxy_cache_report",
    "bootstrap_completion",
)


def _bootstrap_trial_spec(project_root: Path) -> dict:
    return _trial_spec(profile="bootstrap", attempt_kind="bootstrap_proxy", project_root=project_root)


def _reserve_bootstrap(ledger: ResearchEventLedger, direction: dict, variant: dict) -> tuple[dict, dict]:
    trial_spec = _bootstrap_trial_spec(ledger.project_root)
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
    return _c2c_strict_evidence_inventory(
        project_root=root,
        attempt=attempt,
        trial_spec=trial_spec,
        comparison_candidate={
            "metrics": {"mean": 1.0, "datasets": {"fake": 1.0}},
            "proxy_screen": {
                "metrics": {"mean": 1.0, "datasets": {"fake": 1.0}},
                "baseline_metrics": {"mean": 0.0, "datasets": {"fake": 0.0}},
            },
        },
        baseline={"mean": 0.0, "datasets": {"fake": 0.0}},
        simulate=True,
    )


def _prepare_s3_outputs(root: Path, *, bootstrap: bool = False) -> tuple[ResearchEventLedger, dict, dict, dict, dict]:
    direction = _direction()
    variant = _variant(direction, 1)
    write_json(root / "literature" / "direction.json", direction)
    write_json(root / "plan" / "variant.json", variant)
    ledger = ResearchEventLedger(root)
    _initialize(ledger, direction, variant)
    if bootstrap:
        attempt, trial_spec = _reserve_bootstrap(ledger, direction, variant)
        attempt = start_attempt_phase(ledger, attempt, "proxy")
        inventory = _bootstrap_inventory(root, attempt, trial_spec)
        completion = _stage_evidence_inventory(
            project_root=root,
            attempt=attempt,
            trial_spec=trial_spec,
            inventory=inventory,
        )
        record_completed_evidence_command(root, ledger, attempt, completion)
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
    assert state["method_tried_history"] == []
    assert state["latest_direction_aggregate"] is None
    assert state["directions"][direction["direction_semantic_hash"]]["status"] == "ACTIVE"
    assert list(state["trial_results"]) == [attempt["attempt_id"]]
    assert state["last_route_outcome"]["next_action"] == "FINISH_RUN"
    assert state["last_route_outcome"]["source"]["attempt_id"] == attempt["attempt_id"]
    assert sum(event["event_type"] == "AttemptFinalized" for event in after_events) == 1


@pytest.mark.parametrize(
    "extra_kind",
    [
        "effective_proxy_policy",
        "proxy_calibration_policy",
        "failure_evidence",
        "resource_probe",
        "resume_evidence",
    ],
)
def test_bootstrap_completion_rejects_noncompletion_authoritative_kind(tmp_path: Path, extra_kind: str) -> None:
    direction = _direction()
    variant = _variant(direction, 1)
    write_json(tmp_path / "literature" / "direction.json", direction)
    write_json(tmp_path / "plan" / "variant.json", variant)
    ledger = ResearchEventLedger(tmp_path)
    _initialize(ledger, direction, variant)
    attempt, trial_spec = _reserve_bootstrap(ledger, direction, variant)
    attempt = start_attempt_phase(ledger, attempt, "proxy")
    completion = _stage_evidence_inventory(
        project_root=tmp_path,
        attempt=attempt,
        trial_spec=trial_spec,
        inventory=_bootstrap_inventory(tmp_path, attempt, trial_spec),
    )
    invalid = deepcopy(completion)
    invalid["entries"].append(
        {
            **deepcopy(invalid["entries"][0]),
            "evidence_id": f"evidence:{extra_kind}",
            "kind": extra_kind,
        }
    )
    before_events = ledger.events()
    before_state = ledger.state()

    with pytest.raises(IntegrityError):
        ledger.complete_attempt(invalid)

    assert ledger.events() == before_events
    assert ledger.state() == before_state


@pytest.mark.parametrize("missing_kind", _BOOTSTRAP_EVIDENCE_KINDS)
def test_bootstrap_missing_preregistered_evidence_is_zero_write(tmp_path: Path, missing_kind: str) -> None:
    direction = _direction()
    variant = _variant(direction, 1)
    write_json(tmp_path / "literature" / "direction.json", direction)
    write_json(tmp_path / "plan" / "variant.json", variant)
    ledger = ResearchEventLedger(tmp_path)
    _initialize(ledger, direction, variant)
    attempt, trial_spec = _reserve_bootstrap(ledger, direction, variant)
    attempt = start_attempt_phase(ledger, attempt, "proxy")
    inventory = _bootstrap_inventory(tmp_path, attempt, trial_spec)
    completion = _stage_evidence_inventory(
        project_root=tmp_path,
        attempt=attempt,
        trial_spec=trial_spec,
        inventory=inventory,
    )
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

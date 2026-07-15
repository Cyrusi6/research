from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from auto_research.domain_contracts import canonical_hash, variant_semantic_hash, variant_spec_hash
from auto_research.evidence import content_addressed_evidence_path, encode_canonical_evidence
from auto_research.failure_validation import canonical_evidence_bytes, evidence_bytes_hash
from auto_research.research_state import IntegrityError, ResearchEventLedger
from test_authoritative_state_machine import (
    _attempt_inputs,
    _complete,
    _completion_evidence,
    _direction,
    _failure_evidence,
    _initialize,
    _reserve,
    _trial_spec,
    _variant,
)
from support.authoritative_evidence import start_attempt_phase


def _reserve_kind(
    ledger: ResearchEventLedger,
    direction: dict,
    variant: dict,
    *,
    profile: str,
    attempt_kind: str,
) -> dict:
    values = _attempt_inputs(variant)
    trial_spec = _trial_spec(profile=profile, attempt_kind=attempt_kind, project_root=ledger.project_root)
    return ledger.reserve_attempt(
        profile=profile,
        direction=direction,
        variant=variant,
        attempt_kind=attempt_kind,
        trial_spec=trial_spec,
        **values,
    )


def _activation_failure_v4(project_root: Path, attempt: dict) -> dict:
    execution = attempt["phase_executions"]["full"]
    producer_run_id = execution["producer_run_id"]
    identity = {
        "attempt_id": attempt["attempt_id"], "producer_run_id": producer_run_id,
        "direction_semantic_hash": attempt["direction_semantic_hash"], "direction_spec_hash": attempt["direction_spec_hash"],
        "variant_semantic_hash": attempt["variant_semantic_hash"], "variant_spec_hash": attempt["variant_spec_hash"],
        "trial_spec_hash": attempt["trial_spec_hash"], "protocol_hash": attempt["protocol_hash"],
        "sample_manifest_hash": attempt["sample_manifest_hash"], "evaluator_hash": attempt["evaluator_hash"],
        "lifecycle_generation": attempt["lifecycle_generation"], "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"], "phase": "full",
        "phase_execution_id": execution["phase_execution_id"], "phase_start_event_id": execution["phase_start_event_id"],
    }
    receipt = {
        "schema_version": "auto_research_command_result_evidence_v1", "evidence_kind": "command_result_evidence",
        "evidence_id": "activation-command-failure", **identity, "command_id": "activation-command-0001",
        "command": ["python", "activation_probe.py"], "working_directory": "runner",
        "started_at": "2026-07-15T00:00:00Z", "finished_at": "2026-07-15T00:00:01Z",
        "command_status": "failed", "exit_code": 2, "stdout_hash": "b" * 64, "stderr_hash": "c" * 64,
    }
    receipt_raw = canonical_evidence_bytes(receipt)
    receipt_hash = evidence_bytes_hash(receipt_raw)
    receipt_path = project_root / "experiment" / "attempts" / attempt["attempt_id"] / producer_run_id / "command_result_evidence" / f"{receipt_hash}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(receipt_raw)
    failure = {
        "schema_version": "auto_research_failure_evidence_v4", "evidence_kind": "failure_evidence",
        "evidence_id": "activation-failure-0001", **identity,
        "cross_references": {"command_result_evidence_hash": receipt_hash},
        "source_state": "FULL_RUNNING", "source_phase": "full", "failure_class": "activation_failure",
        "command_status": "failed", "exit_code": 2, "reason": "activation failed",
        "observed_at": "2026-07-15T00:00:01Z", "log_hash": "c" * 64,
    }
    failure_raw = canonical_evidence_bytes(failure)
    failure_hash = evidence_bytes_hash(failure_raw)
    failure_path = project_root / content_addressed_evidence_path(
        attempt_id=attempt["attempt_id"], producer_run_id=producer_run_id,
        evidence_kind="failure_evidence", content_hash=failure_hash,
    )
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_bytes(failure_raw)
    return failure


@pytest.mark.parametrize(
    ("profile", "attempt_kind", "allowed", "consumes", "reserved"),
    [
        ("bootstrap", "bootstrap_proxy", True, False, False),
        ("bootstrap", "proxy", False, False, False),
        ("bootstrap", "full", False, False, False),
        ("bootstrap", "proxy_full", False, False, False),
        ("standard", "proxy", True, True, True),
        ("standard", "full", True, True, True),
        ("standard", "proxy_full", True, True, True),
        ("standard", "bootstrap_proxy", False, False, False),
    ],
)
def test_profile_attempt_kind_budget_matrix(
    tmp_path: Path,
    profile: str,
    attempt_kind: str,
    allowed: bool,
    consumes: bool,
    reserved: bool,
) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    before_events = len(ledger.events())
    before_state = ledger.state()

    if not allowed:
        with pytest.raises(IntegrityError, match="profile|attempt kind|bootstrap|standard"):
            _reserve_kind(ledger, direction, variant, profile=profile, attempt_kind=attempt_kind)
        assert len(ledger.events()) == before_events
        assert ledger.state() == before_state
        return

    attempt = _reserve_kind(ledger, direction, variant, profile=profile, attempt_kind=attempt_kind)
    assert attempt["consumes_direction_budget"] is consumes
    assert attempt["reserved_slot"] is reserved


def test_coordinate_nonce_and_identity_changes_do_not_create_new_science(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    original = _variant(direction, 1)
    _initialize(ledger, direction, original)
    _complete(ledger, _reserve(ledger, direction, original), outcome="rejected")

    duplicate = deepcopy(original)
    duplicate["variant_id"] = "direction-variant-999"
    duplicate["lineage"] = {
        **duplicate["lineage"],
        "iteration": 999,
        "s2_run_id": "nonce-only-run",
    }
    duplicate["variation_coordinates"] = deepcopy(duplicate["variation_coordinates"])
    duplicate["variation_coordinates"]["intervention"]["variant_nonce"] = "nonce-999"
    duplicate["variant_semantic_hash"] = variant_semantic_hash(duplicate)
    duplicate["variant_spec_hash"] = variant_spec_hash(duplicate)

    assert duplicate["variant_semantic_hash"] == original["variant_semantic_hash"]
    before_events = len(ledger.events())
    with pytest.raises((IntegrityError, ValueError), match="duplicate"):
        ledger.plan_variant(duplicate)
    assert len(ledger.events()) == before_events


def test_proxy_completed_requires_completed_proxy_phase(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve_kind(ledger, direction, variant, profile="standard", attempt_kind="proxy_full")
    attempt = start_attempt_phase(ledger, attempt, "proxy")
    before_events = len(ledger.events())

    with pytest.raises(IntegrityError, match="PROXY_COMPLETED|phase"):
        ledger.transition_attempt(attempt["attempt_id"], "PROXY_COMPLETED")
    assert len(ledger.events()) == before_events


def test_terminal_phase_cannot_regress_when_entering_full(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve_kind(ledger, direction, variant, profile="standard", attempt_kind="proxy_full")
    attempt = start_attempt_phase(ledger, attempt, "proxy")
    before_events = len(ledger.events())

    with pytest.raises(IntegrityError, match="phase|regress|monotonic"):
        ledger.start_full_phase(attempt["attempt_id"], phase_execution_id="phase-full-forged", producer_run_id="producer-full-forged")
    assert len(ledger.events()) == before_events


def test_ready_attempt_cannot_finalize(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    completion = {
        "schema_version": "auto_research_completion_evidence_v2",
        "attempt_id": attempt["attempt_id"],
        "trial_spec_hash": attempt["trial_spec_hash"],
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "entries": [],
    }
    _assert_trial_rejected_without_writes(ledger, completion, match="READY|execution state|cannot finalize")


def test_failed_phase_cannot_be_overwritten_by_finalization(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    running = start_attempt_phase(ledger, attempt, "full")
    completion = _valid_completion(ledger, running)
    ledger.disposition_failure(_activation_failure_v4(tmp_path, running))
    _assert_trial_rejected_without_writes(ledger, completion, match="phase|FAILED|completed|execution state|IMPLEMENTATION_REPAIR")


def test_full_trial_requires_full_execution_state(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve_kind(ledger, direction, variant, profile="standard", attempt_kind="proxy_full")
    attempt = start_attempt_phase(ledger, attempt, "proxy")
    completion = {
        "schema_version": "auto_research_completion_evidence_v2",
        "attempt_id": attempt["attempt_id"], "trial_spec_hash": attempt["trial_spec_hash"],
        "lifecycle_generation": attempt["lifecycle_generation"], "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"], "entries": [],
    }
    _assert_trial_rejected_without_writes(ledger, completion, match="PROXY_RUNNING|full|phase|execution state")


def _valid_completion(ledger: ResearchEventLedger, attempt: dict) -> dict:
    return _completion_evidence(ledger, attempt, outcome="accepted")


def _rewrite_completion_artifact(ledger: ResearchEventLedger, completion: dict, mutate) -> None:
    entry = completion["entries"][0]
    payload = json.loads((ledger.project_root / entry["relative_path"]).read_text(encoding="utf-8"))
    mutate(payload)
    raw = encode_canonical_evidence(payload)
    digest = hashlib.sha256(raw).hexdigest()
    relative_path = content_addressed_evidence_path(
        attempt_id=entry["attempt_id"],
        producer_run_id=entry["producer_run_id"],
        evidence_kind=entry["kind"],
        content_hash=digest,
    )
    path = ledger.project_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    entry["relative_path"] = relative_path
    entry["content_hash"] = digest


def _assert_trial_rejected_without_writes(
    ledger: ResearchEventLedger,
    completion: dict,
    *,
    match: str,
) -> None:
    before_events = ledger.events()
    before_state = ledger.state()
    before_trial = ledger.trial_path.read_bytes() if ledger.trial_path.exists() else None

    with pytest.raises((IntegrityError, ValueError), match=match):
        ledger.complete_attempt(completion)

    assert ledger.events() == before_events
    assert ledger.state() == before_state
    assert (ledger.trial_path.read_bytes() if ledger.trial_path.exists() else None) == before_trial


@pytest.mark.parametrize(
    "forgery",
    [
        "unregistered_seed",
        "sample_manifest_hash",
        "evaluator_hash",
        "dataset_contract",
        "duplicate_observation",
        "unregistered_raw_artifact",
        "failed_command",
        "partial_command",
        "resource_paused_command",
        "integrity_blocked_command",
        "missing_baseline_role",
    ],
)
def test_ledger_independently_rejects_forged_trial_result(
    tmp_path: Path,
    forgery: str,
) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    attempt = start_attempt_phase(ledger, attempt, "full")
    completion = _valid_completion(ledger, attempt)
    assert ledger.validate_trial_precommit(completion)["outcome_classification"] == "accepted"

    if forgery == "unregistered_seed":
        _rewrite_completion_artifact(ledger, completion, lambda payload: payload["rows"][1].__setitem__("seed", 999))
    elif forgery == "sample_manifest_hash":
        _rewrite_completion_artifact(ledger, completion, lambda payload: payload["rows"][1].__setitem__("sample_manifest_hash", "a" * 64))
    elif forgery == "evaluator_hash":
        _rewrite_completion_artifact(ledger, completion, lambda payload: payload["rows"][1].__setitem__("evaluator_hash", "b" * 64))
    elif forgery == "dataset_contract":
        _rewrite_completion_artifact(
            ledger,
            completion,
            lambda payload: [row.__setitem__("dataset_id", "unregistered-dataset") for row in payload["rows"]],
        )
    elif forgery == "duplicate_observation":
        _rewrite_completion_artifact(ledger, completion, lambda payload: payload["rows"].append(deepcopy(payload["rows"][1])))
    elif forgery == "unregistered_raw_artifact":
        completion["entries"][0]["content_hash"] = "c" * 64
    elif forgery in {
        "failed_command",
        "partial_command",
        "resource_paused_command",
        "integrity_blocked_command",
    }:
        _rewrite_completion_artifact(
            ledger,
            completion,
            lambda payload: payload["rows"][1].__setitem__("command_status", forgery.removesuffix("_command")),
        )
    elif forgery == "missing_baseline_role":
        _rewrite_completion_artifact(
            ledger,
            completion,
            lambda payload: payload.__setitem__("rows", [row for row in payload["rows"] if row["role"] != "baseline"]),
        )
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(f"unknown forgery {forgery}")

    _assert_trial_rejected_without_writes(
        ledger,
        completion,
        match="seed|hash|dataset|duplicate|artifact|command|status|role|coverage|observation",
    )

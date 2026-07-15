from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from auto_research.domain_contracts import canonical_hash, variant_semantic_hash, variant_spec_hash
from auto_research.evidence import content_addressed_evidence_path, encode_canonical_evidence
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
from support.authoritative_evidence import (
    build_quantitative_completion,
    record_completed_evidence_command,
    start_attempt_phase,
)


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
    staged_attempt = deepcopy(attempt)
    staged_attempt["phase_executions"]["full"] = _schema_only_phase_execution(attempt, "full", "ready")
    completion = _schema_valid_completion(ledger, staged_attempt, "full")
    _assert_trial_rejected_without_writes(ledger, completion, match="READY|execution state|cannot finalize")


def test_failed_phase_cannot_be_overwritten_by_finalization(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    running = start_attempt_phase(ledger, attempt, "full")
    completion = _valid_completion(ledger, running)
    ledger.disposition_failure(_failure_evidence(ledger, running, "activation_failure"))
    _assert_trial_rejected_without_writes(ledger, completion, match="phase|FAILED|completed|execution state|IMPLEMENTATION_REPAIR")


def test_full_trial_requires_full_execution_state(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve_kind(ledger, direction, variant, profile="standard", attempt_kind="proxy_full")
    attempt = start_attempt_phase(ledger, attempt, "proxy")
    staged_attempt = deepcopy(attempt)
    staged_attempt["phase_executions"]["full"] = _schema_only_phase_execution(attempt, "full", "proxy-state")
    completion = _schema_valid_completion(ledger, staged_attempt, "full")
    _assert_trial_rejected_without_writes(ledger, completion, match="PROXY_RUNNING|full|phase|execution state")


def _valid_completion(ledger: ResearchEventLedger, attempt: dict) -> dict:
    return _completion_evidence(ledger, attempt, outcome="accepted")


def _schema_only_phase_execution(attempt: dict, phase: str, suffix: str) -> dict:
    phase_contract = next(item for item in attempt["frozen_trial_spec"]["phase_contracts"] if item["phase"] == phase)
    return {
        "phase_execution_id": f"{phase}-{suffix}-schema-attack",
        "phase_start_event_id": f"evt-{phase}-{suffix}-schema-attack",
        "producer_run_id": f"producer-{phase}-{suffix}-schema-attack",
        "command_plan_hash": phase_contract["command_plan_hash"],
    }


def _schema_valid_completion(ledger: ResearchEventLedger, attempt: dict, phase: str) -> dict:
    return build_quantitative_completion(
        ledger.project_root,
        attempt,
        role_values={"baseline": 0.5, "candidate": 0.7},
        dataset_id="fake",
        metric_id="accuracy",
        seed=1,
        phase=phase,
    )


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
    record_completed_evidence_command(ledger.project_root, ledger, attempt, completion)
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

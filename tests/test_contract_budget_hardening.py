from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from auto_research.domain_contracts import (
    EVIDENCE_MANIFEST_SCHEMA_VERSION,
    EXECUTION_OBSERVATION_SCHEMA_VERSION,
    canonical_hash,
    classify_trial_result,
    variant_semantic_hash,
    variant_spec_hash,
)
from auto_research.research_state import IntegrityError, ResearchEventLedger
from test_authoritative_state_machine import (
    _attempt_inputs,
    _complete,
    _direction,
    _failure_evidence,
    _initialize,
    _reserve,
    _trial_spec,
    _variant,
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
    trial_spec = _trial_spec(profile=profile, attempt_kind=attempt_kind)
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
    ledger.transition_attempt(attempt["attempt_id"], "PROXY_RUNNING", phase="proxy", phase_state="RUNNING")
    before_events = len(ledger.events())

    with pytest.raises(IntegrityError, match="proxy.*COMPLETED|phase"):
        ledger.transition_attempt(attempt["attempt_id"], "PROXY_COMPLETED")
    assert len(ledger.events()) == before_events


def test_terminal_phase_cannot_regress_when_entering_full(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve_kind(ledger, direction, variant, profile="standard", attempt_kind="proxy_full")
    ledger.transition_attempt(attempt["attempt_id"], "PROXY_RUNNING", phase="proxy", phase_state="RUNNING")
    ledger.transition_attempt(attempt["attempt_id"], "PROXY_COMPLETED", phase="proxy", phase_state="COMPLETED")
    before_events = len(ledger.events())

    with pytest.raises(IntegrityError, match="phase|regress|monotonic"):
        ledger.transition_attempt(attempt["attempt_id"], "FULL_RUNNING", phase="proxy", phase_state="PENDING")
    assert len(ledger.events()) == before_events


def test_ready_attempt_cannot_finalize(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    trial = _valid_trial(ledger, attempt)
    _assert_trial_rejected_without_writes(ledger, trial, match="READY|execution state|cannot finalize")


def test_failed_phase_cannot_be_overwritten_by_finalization(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    ledger.transition_attempt(attempt["attempt_id"], "FULL_RUNNING", phase="full", phase_state="RUNNING")
    running = ledger.state()["attempts"][attempt["attempt_id"]]
    ledger.disposition_failure(_failure_evidence(ledger, running, "activation_failure"))
    trial = _valid_trial(ledger, ledger.state()["attempts"][attempt["attempt_id"]])
    _assert_trial_rejected_without_writes(ledger, trial, match="phase|FAILED|completed|execution state|IMPLEMENTATION_REPAIR")


def test_full_trial_requires_full_execution_state(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve_kind(ledger, direction, variant, profile="standard", attempt_kind="proxy_full")
    ledger.transition_attempt(attempt["attempt_id"], "PROXY_RUNNING", phase="proxy", phase_state="RUNNING")
    trial = _valid_trial(ledger, ledger.state()["attempts"][attempt["attempt_id"]])
    _assert_trial_rejected_without_writes(ledger, trial, match="full|phase|execution state")


def _valid_trial(ledger: ResearchEventLedger, attempt: dict) -> dict:
    artifact = ledger.project_root / "experiment" / "raw" / f"{attempt['attempt_id']}.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('{"schema_version":"auto_research_main_results_v1"}\n', encoding="utf-8")
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    phase = "proxy" if attempt["attempt_kind"] in {"proxy", "bootstrap_proxy"} else "full"
    observations = [
        {
            "schema_version": EXECUTION_OBSERVATION_SCHEMA_VERSION,
            "observation_id": f"obs:{attempt['attempt_id'][:8]}:{role}:1",
            "phase": phase,
            "role": role,
            "command_status": "completed",
            "dataset_id": "fake",
            "metric_id": "accuracy",
            "metric_value": value,
            "sample_manifest_hash": attempt["sample_manifest_hash"],
            "evaluator_hash": attempt["evaluator_hash"],
            "seed": 1,
            "raw_artifact_path": str(artifact.relative_to(ledger.project_root)),
            "raw_artifact_hash": artifact_hash,
        }
        for role, value in (("baseline", 0.5), ("candidate", 0.7))
    ]
    evidence_manifest = {
        "schema_version": EVIDENCE_MANIFEST_SCHEMA_VERSION,
        "trial_spec_hash": attempt["trial_spec_hash"],
        "attempt_id": attempt["attempt_id"],
        "entries": [{
            "evidence_id": f"evidence:{attempt['attempt_id'][:8]}:main",
            "kind": "main_results",
            "relative_path": str(artifact.relative_to(ledger.project_root)),
            "content_hash": artifact_hash,
            "schema_version": "auto_research_main_results_v1",
            "attempt_id": attempt["attempt_id"],
            "variant_spec_hash": attempt["variant_spec_hash"],
            "trial_spec_hash": attempt["trial_spec_hash"],
            "cross_references": {},
        }],
    }
    return classify_trial_result(
        attempt=attempt,
        trial_spec=attempt["frozen_trial_spec"],
        observations=observations,
        raw_artifacts={str(artifact.relative_to(ledger.project_root)): artifact_hash},
        evidence_manifest=evidence_manifest,
    )


def _assert_trial_rejected_without_writes(
    ledger: ResearchEventLedger,
    trial: dict,
    *,
    match: str,
) -> None:
    before_events = ledger.events()
    before_state = ledger.state()
    before_trial = ledger.trial_path.read_bytes() if ledger.trial_path.exists() else None

    with pytest.raises((IntegrityError, ValueError), match=match):
        ledger.complete_attempt(trial)

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
    ledger.transition_attempt(attempt["attempt_id"], "FULL_RUNNING", phase="full", phase_state="RUNNING")
    trial = _valid_trial(ledger, ledger.state()["attempts"][attempt["attempt_id"]])

    if forgery == "unregistered_seed":
        trial["observations"][1]["seed"] = 999
    elif forgery == "sample_manifest_hash":
        trial["observations"][1]["sample_manifest_hash"] = "a" * 64
    elif forgery == "evaluator_hash":
        trial["observations"][1]["evaluator_hash"] = "b" * 64
    elif forgery == "dataset_contract":
        for observation in trial["observations"]:
            observation["dataset_id"] = "unregistered-dataset"
        trial["required_datasets"] = ["unregistered-dataset"]
        trial["observed_datasets"] = ["unregistered-dataset"]
    elif forgery == "duplicate_observation":
        trial["observations"].append(deepcopy(trial["observations"][1]))
    elif forgery == "unregistered_raw_artifact":
        trial["observations"][1]["raw_artifact_hash"] = "c" * 64
    elif forgery in {
        "failed_command",
        "partial_command",
        "resource_paused_command",
        "integrity_blocked_command",
    }:
        trial["observations"][1]["command_status"] = forgery.removesuffix("_command")
    elif forgery == "missing_baseline_role":
        trial["observations"] = [item for item in trial["observations"] if item["role"] != "baseline"]
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(f"unknown forgery {forgery}")

    _assert_trial_rejected_without_writes(
        ledger,
        trial,
        match="seed|hash|dataset|duplicate|artifact|command|status|role|coverage|observation",
    )

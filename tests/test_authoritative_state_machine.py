from __future__ import annotations

import hashlib
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest

from auto_research.domain_contracts import (
    TRIAL_SPEC_SCHEMA_VERSION,
    build_direction_spec,
    build_variant_spec,
    canonical_hash,
    direction_spec_hash,
    implementation_hash,
    validate_contract,
    validate_trial_result,
    validate_variant_identity,
    variant_semantic_hash,
    variant_spec_hash,
)
from auto_research.evidence import content_addressed_evidence_path, encode_canonical_evidence
from support.authoritative_evidence import build_quantitative_completion
from auto_research.research_state import FAILURE_EVIDENCE_SCHEMA_VERSION, IntegrityError, ResearchEventLedger
from auto_research.utils import write_json


def _direction() -> dict:
    return build_direction_spec(
        {
            "direction_id": "direction-alpha",
            "research_question": "Does mediator-aware routing improve the benchmark outcome?",
            "mechanism_invariants": {
                "causal_hypothesis": "Changing mediator-aware routing improves the target metric.",
                "target_mediator": "routing_quality",
                "invariants": ["same mediator", "same benchmark", "same causal mechanism"],
            },
            "falsification_conditions": ["routing quality does not change", "the target metric does not improve"],
            "support_claim_ids": ["support-1"],
            "counter_claim_ids": ["counter-1"],
            "implementation_surface_ids": ["src/router.py"],
            "metric_signature": {"primary": "accuracy", "direction": "increase"},
            "benchmark_contract_hash": canonical_hash({"datasets": ["fake"]}),
            "variant_space": {
                "mutable_axes": ["intervention"],
                "immutable_axes": ["benchmark", "mediator"],
                "forbidden_combinations": [{"intervention": "forbidden"}],
            },
            "s2_entry_conditions": ["S1 gate passes"],
            "return_to_s1_conditions": ["five outcomes reject the direction"],
            "lineage": {"s1_run_id": "s1-run", "iteration": 1, "input_manifest_hash": canonical_hash({"input": 1})},
        }
    )


def _variant(direction: dict, index: int, feedback: list[str] | None = None) -> dict:
    return build_variant_spec(
        direction,
        {
            "variant_id": f"variant-{index}",
            "variation_coordinates": {"intervention": {"operation": f"operation-{index}", "strength": index}},
            "intervention": {
                "summary": f"Apply operation {index}",
                "algorithm_operations": [f"operation-{index}"],
                "configuration": {"strength": index},
            },
            "hypothesis": f"Operation {index} improves accuracy.",
            "null_hypothesis": f"Operation {index} does not improve accuracy.",
            "alternative_hypothesis": f"Operation {index} improves accuracy.",
            "controlled_variables": {"dataset": "fake", "seed_policy": "fixed"},
            "nuisance_variables": ["gpu_noise"],
            "implementation_surface_ids": ["src/router.py"],
            "expected_metric_signature": {"primary": "accuracy", "direction": "increase"},
            "falsification_conditions": ["accuracy does not improve"],
            "ablation": {"switch": f"disable_operation_{index}"},
            "resource_budget": {"max_wall_seconds": 60, "max_retries": 2},
            "failure_routing": {"implementation": "REPAIR_IMPLEMENTATION", "method": "PROPOSE_NEXT_VARIANT"},
            "lineage": {
                "s2_run_id": f"s2-run-{index}",
                "iteration": index,
                "direction_spec_hash": direction["direction_spec_hash"],
                "feedback_from_attempt_ids": feedback or [],
            },
        },
    )



def _trial_spec(attempt: dict | None = None, *, profile: str = "standard", attempt_kind: str | None = None) -> dict:
    from support.authoritative_evidence import upgrade_trial_spec_v4
    return upgrade_trial_spec_v4(_trial_spec_legacy(attempt, profile=profile, attempt_kind=attempt_kind))

def _initialize(ledger: ResearchEventLedger, direction: dict, variant: dict) -> None:
    ledger.select_direction(direction)
    ledger.plan_variant(variant, feedback_from_attempt_ids=(variant.get("lineage") or {}).get("feedback_from_attempt_ids") or [])


def _trial_spec_legacy(attempt: dict | None = None, *, profile: str = "standard", attempt_kind: str | None = None) -> dict:
    kind = attempt_kind or ((attempt or {}).get("attempt_kind")) or ("bootstrap_proxy" if profile == "bootstrap" else "full")
    phase = "proxy" if kind in {"proxy", "bootstrap_proxy"} else "full"
    runtime = {"batch": 1, "device": "cpu"}
    sample_id = canonical_hash({"sample": "fake-1"})
    sample_dataset = {
        "dataset_id": "fake", "source_revision": "synthetic-v1", "split": "test",
        "sample_count": 1, "ordered_sample_ids": [sample_id],
    }
    sample_dataset["content_hash"] = canonical_hash({
        "dataset_id": "fake", "source_revision": "synthetic-v1", "split": "test",
        "ordered_sample_ids": [sample_id],
    })
    sample_manifest = {
        "schema_version": "auto_research_sample_manifest_v1", "manifest_id": "fake-v1",
        "provenance_mode": "synthetic", "datasets": [sample_dataset],
        "artifact_path": "plan/sample_manifest.json",
    }
    sample_manifest["artifact_hash"] = canonical_hash(sample_manifest)
    evaluator = {
        "schema_version": "auto_research_evaluator_provenance_v1", "provenance_mode": "synthetic",
        "evaluator_id": "fake-evaluator", "source_digest": canonical_hash({"source": "fake"}),
        "config_hash": canonical_hash({"metric": "accuracy"}),
        "dependency_digest": canonical_hash({"dependencies": []}),
    }
    return {
        "schema_version": TRIAL_SPEC_SCHEMA_VERSION,
        "protocol": {
            "protocol_id": f"{phase}-v1",
            "required_phases": [phase],
            "terminal_phases": [phase],
            "proxy_terminal_allowed": phase == "proxy",
            "aggregation": "mean",
        },
        "sample_manifest": sample_manifest,
        "datasets": [{"dataset_id": "fake", "split": "test", "sample_count": 1, "sample_hash": sample_dataset["content_hash"]}],
        "metrics": [{"metric_id": "accuracy", "objective": "maximize", "aggregation": "mean", "role": "primary"}],
        "primary_metric_id": "accuracy",
        "statistical_testing": {"method": "none", "seeds": [1], "require_complete_seed_coverage": True},
        "required_roles": ["baseline", "candidate"],
        "acceptance_constraints": [{
            "constraint_id": "primary-delta",
            "kind": "minimum_mean_delta",
            "hard": True,
            "metric_id": "accuracy",
            "threshold": 0.05,
            "objective": "maximize",
        }],
        "execution_contract": {
            "runtime_config": runtime,
            "runtime_config_hash": canonical_hash(runtime),
            "evaluator_provenance": evaluator,
            "evaluator_hash": canonical_hash(evaluator),
            "command_contract_hash": canonical_hash({"command": phase}),
        },
        "required_artifacts": ["main_results"],
        "evidence_requirements": [{
            "requirement_id": "main-results",
            "kind": "main_results",
            "required": True,
            "applicable_phases": [phase],
            "schema_version": "auto_research_main_results_v2",
        }],
    }


def _attempt_inputs(variant: dict) -> dict:
    return {"implementation_hash": implementation_hash(
        frozen_patch={"variant": variant["variant_id"]},
        files={"src/router.py": variant["variant_id"]},
        manifest={"v": 1},
    )}


def _reserve(ledger: ResearchEventLedger, direction: dict, variant: dict, *, profile: str = "standard") -> dict:
    values = _attempt_inputs(variant)
    kind = "bootstrap_proxy" if profile == "bootstrap" else "full"
    trial_spec = _trial_spec(profile=profile, attempt_kind=kind)
    write_json(ledger.project_root / "plan" / "trial_spec.json", trial_spec)
    return ledger.reserve_attempt(
        profile=profile,
        direction=direction,
        variant=variant,
        attempt_kind=kind,
        trial_spec=trial_spec,
        **values,
    )


def _failure_evidence(ledger: ResearchEventLedger, attempt: dict, failure_class: str, *, suffix: str = "1") -> dict:
    from support.authoritative_evidence import start_attempt_phase
    aliases = {"resource_paused": "resource_pause", "integrity": "integrity_failure", "activation_failed": "activation_failure"}
    failure_class = aliases.get(failure_class, failure_class)
    is_resource = failure_class in {"resource_pause", "oom_retry"}
    current = ledger.state()["attempts"][attempt["attempt_id"]]
    execution_phase = "proxy" if current["attempt_kind"] in {"proxy", "bootstrap_proxy", "proxy_full"} else "full"
    if not isinstance(current["phase_executions"][execution_phase], dict) and failure_class != "implementation_failure":
        current = start_attempt_phase(ledger, current, execution_phase)
    attempt = current
    producer_run_id = f"failure-producer-{attempt['lifecycle_generation']}-{suffix}"
    phase = next((name for name, status in attempt["phases"].items() if status == "RUNNING"), None)
    if failure_class == "implementation_failure":
        phase = "implementation"
    elif failure_class == "activation_failure" or phase is None:
        phase = "activation"
    if is_resource:
        artifact_payload = {
            "schema_version": "auto_research_resource_probe_evidence_v2",
            "evidence_kind": "resource_probe",
            "evidence_id": f"resource-probe-{attempt['lifecycle_generation']}-{suffix}",
            "attempt_id": attempt["attempt_id"],
            "producer_run_id": producer_run_id,
            "direction_semantic_hash": attempt["direction_semantic_hash"],
            "direction_spec_hash": attempt["direction_spec_hash"],
            "variant_semantic_hash": attempt["variant_semantic_hash"],
            "variant_spec_hash": attempt["variant_spec_hash"],
            "trial_spec_hash": attempt["trial_spec_hash"],
            "protocol_hash": attempt["protocol_hash"],
            "sample_manifest_hash": attempt["sample_manifest_hash"],
            "evaluator_hash": attempt["evaluator_hash"],
            "cross_references": {},
            "resource_type": "system_memory",
            "resource_id": "memory-0",
            "required_capacity": 10.0,
            "observed_capacity": 1.0,
            "unit": "bytes",
            "probe_status": "insufficient",
            "observed_at": "2026-01-01T00:00:00Z",
            "lifecycle_generation": attempt["lifecycle_generation"],
            "implementation_hash": attempt["implementation_hash"],
            "attempt_input_hash": attempt["attempt_input_hash"],
            "phase": execution_phase,
            "phase_execution_id": attempt["phase_executions"][execution_phase]["phase_execution_id"],
            "phase_start_event_id": attempt["phase_executions"][execution_phase]["phase_start_event_id"],
        }
        artifact_kind = "resource_probe"
    else:
        artifact_payload = {
            "attempt_id": attempt["attempt_id"],
            "failure_class": failure_class,
            "producer_run_id": producer_run_id,
            "suffix": suffix,
        }
        artifact_kind = "failure_evidence"
    artifact_bytes = encode_canonical_evidence(artifact_payload)
    artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
    relative_path = content_addressed_evidence_path(
        attempt_id=attempt["attempt_id"],
        producer_run_id=producer_run_id,
        evidence_kind=artifact_kind,
        content_hash=artifact_hash,
    )
    artifact = ledger.project_root / relative_path
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(artifact_bytes)
    execution = attempt["phase_executions"].get(execution_phase) or {
        "phase_execution_id": f"implementation-{attempt['attempt_id'][:12]}",
        "phase_start_event_id": next(event["event_id"] for event in ledger.events() if event["event_type"] == "AttemptReserved" and event["payload"]["attempt"]["attempt_id"] == attempt["attempt_id"]),
    }
    return {
        "schema_version": FAILURE_EVIDENCE_SCHEMA_VERSION,
        "evidence_kind": "failure_evidence",
        "evidence_id": f"failure-evidence-{attempt['lifecycle_generation']}-{suffix}",
        "attempt_id": attempt["attempt_id"],
        "producer_run_id": producer_run_id,
        "direction_semantic_hash": attempt["direction_semantic_hash"],
        "direction_spec_hash": attempt["direction_spec_hash"],
        "variant_semantic_hash": attempt["variant_semantic_hash"],
        "variant_spec_hash": attempt["variant_spec_hash"],
        "trial_spec_hash": attempt["trial_spec_hash"],
        "protocol_hash": attempt["protocol_hash"],
        "sample_manifest_hash": attempt["sample_manifest_hash"],
        "evaluator_hash": attempt["evaluator_hash"],
        "cross_references": {},
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "phase": phase,
        "phase_execution_id": execution["phase_execution_id"],
        "phase_start_event_id": execution["phase_start_event_id"],
        "source_state": attempt["state"],
        "source_phase": phase,
        "failure_class": failure_class,
        "command_status": "resource_paused" if is_resource else ("integrity_blocked" if failure_class in {"integrity_failure", "safety_failure"} else "failed"),
        "exit_code": 137 if is_resource else 1,
        "reason": f"verified {failure_class}",
        "observed_at": "2026-01-01T00:00:00Z",
        "log_hash": artifact_hash,
    }


def _completion_evidence(
    ledger: ResearchEventLedger,
    attempt: dict,
    *,
    outcome: str,
) -> dict:
    from support.authoritative_evidence import start_attempt_phase
    phase = "proxy" if attempt["attempt_kind"] == "bootstrap_proxy" else "full"
    current = ledger.state()["attempts"][attempt["attempt_id"]]
    if not isinstance(current["phase_executions"][phase], dict):
        current = start_attempt_phase(ledger, current, phase)
    candidate = 0.7 if outcome == "accepted" else 0.51
    return build_quantitative_completion(
        ledger.project_root,
        current,
        role_values={"baseline": 0.5, "candidate": candidate},
        dataset_id="fake",
        metric_id="accuracy",
        seed=1,
        phase=phase,
    )


def _complete(
    ledger: ResearchEventLedger,
    attempt: dict,
    *,
    outcome: str,
    evaluable: bool = True,
    failure: str | None = None,
):
    if not evaluable:
        current = ledger.state()["attempts"][attempt["attempt_id"]]
        return ledger.disposition_failure(_failure_evidence(ledger, current, failure or "implementation_failure"))
    completion = _completion_evidence(ledger, attempt, outcome=outcome)
    return ledger.complete_attempt(completion)


def test_five_outcomes_keep_direction_identity_and_never_create_sixth(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    attempts = []
    for index in range(1, 6):
        variant = _variant(direction, index)
        _initialize(ledger, direction, variant)
        attempt = _reserve(ledger, direction, variant)
        attempts.append(attempt)
        _, route = _complete(ledger, attempt, outcome="accepted")
        assert attempt["direction_spec_hash"] == direction["direction_spec_hash"]
        assert route["next_action"] == ("PROPOSE_NEXT_VARIANT" if index < 5 else "FINISH_DIRECTION")
    state = ledger.state()
    budget = state["directions"][direction["direction_semantic_hash"]]["budget"]
    assert budget == {"target": 5, "reserved": 0, "consumed": 5}
    assert len({item["variant_semantic_hash"] for item in state["method_tried_history"]}) == 5
    with pytest.raises(IntegrityError, match="closed direction"):
        ledger.select_direction(direction, event_id="direction:reopen")


@pytest.mark.parametrize("adapter", ["generic", "c2c"])
def test_all_rejected_standard_direction_starts_new_direction_without_s2_deadlock(tmp_path: Path, adapter: str) -> None:
    ledger = ResearchEventLedger(tmp_path / adapter)
    direction = _direction()
    final_route = None
    for index in range(1, 6):
        variant = _variant(direction, index)
        _initialize(ledger, direction, variant)
        attempt = _reserve(ledger, direction, variant)
        _, final_route = _complete(ledger, attempt, outcome="rejected")

    assert final_route and final_route["next_action"] == "START_NEW_DIRECTION"
    state = ledger.state()
    assert direction["direction_semantic_hash"] in state["excluded_direction_semantic_hashes"]
    with pytest.raises(IntegrityError, match="closed direction"):
        ledger.select_direction(direction, event_id=f"direction:reopen:{adapter}")

    payload = deepcopy(direction)
    payload.pop("schema_version", None)
    payload.pop("direction_semantic_hash", None)
    payload.pop("direction_spec_hash", None)
    payload["direction_id"] = f"direction-beta-{adapter}"
    payload["research_question"] = f"Does a new {adapter} mediator improve the benchmark outcome?"
    payload["mechanism_invariants"] = {
        **payload["mechanism_invariants"],
        "causal_hypothesis": f"A distinct {adapter} intervention changes a new mediator.",
        "target_mediator": f"{adapter}_new_mediator",
    }
    payload["lineage"] = {**payload["lineage"], "iteration": 2, "s1_run_id": f"s1-{adapter}-2"}
    new_direction = build_direction_spec(payload)
    ledger.select_direction(new_direction)
    new_variant = _variant(new_direction, 1)
    ledger.plan_variant(new_variant)
    new_attempt = _reserve(ledger, new_direction, new_variant)
    assert new_attempt["direction_semantic_hash"] != direction["direction_semantic_hash"]
    assert ledger.state()["directions"][new_direction["direction_semantic_hash"]]["budget"] == {"target": 5, "reserved": 1, "consumed": 0}


def test_same_scientific_variant_with_new_id_and_lineage_is_duplicate(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    first = _variant(direction, 1)
    _initialize(ledger, direction, first)
    _complete(ledger, _reserve(ledger, direction, first), outcome="rejected")
    duplicate = deepcopy(first)
    duplicate["variant_id"] = "renamed"
    duplicate["lineage"] = {**duplicate["lineage"], "iteration": 999, "s2_run_id": "new-run"}
    duplicate.pop("variant_spec_hash")
    duplicate["variant_spec_hash"] = variant_spec_hash(duplicate)
    assert duplicate["variant_semantic_hash"] == first["variant_semantic_hash"]
    ledger.select_direction(direction)
    with pytest.raises((IntegrityError, ValueError), match="duplicate"):
        ledger.plan_variant(duplicate)


def test_event_id_idempotency_and_conflict(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    first, _ = ledger.append("AuditMarker", {"index": 1}, event_id="audit:one")
    repeated, _ = ledger.append("AuditMarker", {"index": 1}, event_id="audit:one")
    assert first == repeated
    assert len(ledger.events()) == 1
    with pytest.raises(IntegrityError, match="conflict"):
        ledger.append("AuditMarker", {"index": 2}, event_id="audit:one")


def test_concurrent_append_has_unique_contiguous_sequences(tmp_path: Path) -> None:
    def append(index: int) -> None:
        ResearchEventLedger(tmp_path).append("AuditMarker", {"index": index}, event_id=f"audit:{index}")

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(append, range(100)))
    events = ResearchEventLedger(tmp_path).events()
    assert [event["sequence"] for event in events] == list(range(1, 101))
    assert len({event["event_id"] for event in events}) == 100


def test_rebuild_rejects_hash_chain_tampering(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    ledger.append("AuditMarker", {"index": 1}, event_id="audit:1")
    ledger.append("AuditMarker", {"index": 2}, event_id="audit:2")
    with sqlite3.connect(ledger.db_path) as connection:
        connection.execute("UPDATE events SET previous_event_hash = ? WHERE sequence = 2", ("f" * 64,))
    with pytest.raises(IntegrityError, match="hash chain"):
        ledger.rebuild()


def test_commit_crash_before_projection_rebuilds_unique_result_and_route(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    def crash_after_finalization() -> None:
        if ledger.events()[-1]["event_type"] == "AttemptFinalized":
            raise RuntimeError("crash after commit")

    ledger.after_commit_hook = crash_after_finalization
    with pytest.raises(RuntimeError, match="crash after commit"):
        _complete(ledger, attempt, outcome="rejected")
    state = ResearchEventLedger(tmp_path).rebuild()
    assert list(state["trial_results"]) == [attempt["attempt_id"]]
    assert state["last_route_outcome"]["source"]["attempt_id"] == attempt["attempt_id"]
    assert state["directions"][direction["direction_semantic_hash"]]["budget"]["consumed"] == 1


def test_trial_identity_mismatch_writes_nothing(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    completion = _completion_evidence(ledger, attempt, outcome="accepted")
    before_events = len(ledger.events())
    before_state = ledger.state()
    completion["trial_spec_hash"] = "f" * 64
    with pytest.raises(IntegrityError, match="TrialSpec identity mismatch"):
        ledger.complete_attempt(completion)
    assert len(ledger.events()) == before_events
    after_state = ledger.state()
    assert after_state["directions"] == before_state["directions"]
    assert after_state["last_route_outcome"] == before_state["last_route_outcome"]


def test_fingerprint_stability_and_sensitivity() -> None:
    direction = _direction()
    variant = _variant(direction, 1)
    reordered = {key: variant[key] for key in reversed(list(variant))}
    assert variant_spec_hash(variant) == variant_spec_hash(reordered)
    changed = deepcopy(variant)
    changed["intervention"]["configuration"]["strength"] = 99
    assert variant_semantic_hash(changed) != variant["variant_semantic_hash"]
    assert variant_spec_hash(changed) != variant["variant_spec_hash"]
    changed_direction = deepcopy(direction)
    changed_direction["research_question"] += " Precisely?"
    assert direction_spec_hash(changed_direction) != direction["direction_spec_hash"]


def test_wrong_lineage_direction_spec_hash_fails() -> None:
    direction = _direction()
    variant = _variant(direction, 1)
    variant["lineage"]["direction_spec_hash"] = "f" * 64
    variant["variant_spec_hash"] = variant_spec_hash(variant)
    with pytest.raises(ValueError, match="lineage.direction_spec_hash"):
        validate_variant_identity(direction, variant)


def test_abandoned_attempt_cannot_revive_after_five_outcomes(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    first_variant = _variant(direction, 1)
    _initialize(ledger, direction, first_variant)
    first_attempt = _reserve(ledger, direction, first_variant)
    ledger.disposition_failure(_failure_evidence(ledger, first_attempt, "activation_failure"))
    ledger.abandon_attempt(first_attempt["attempt_id"], reason="replace non-evaluable implementation")
    for index in range(2, 7):
        variant = _variant(direction, index)
        _initialize(ledger, direction, variant)
        _complete(ledger, _reserve(ledger, direction, variant), outcome="rejected")
    assert ledger.state()["directions"][direction["direction_semantic_hash"]]["budget"]["consumed"] == 5
    with pytest.raises(IntegrityError, match="already finalized|cannot finalize|no reserved slot|illegal attempt transition"):
        _complete(ledger, first_attempt, outcome="accepted")


def test_illegal_transition_and_terminal_transition_write_no_event(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    count = len(ledger.events())
    with pytest.raises(IntegrityError, match="cannot enter|illegal attempt transition"):
        ledger.transition_attempt(attempt["attempt_id"], "METHOD_COMPLETED")
    assert len(ledger.events()) == count
    completed, _ = _complete(ledger, attempt, outcome="accepted")
    count = len(ledger.events())
    with pytest.raises(IntegrityError, match="cannot enter|illegal attempt transition"):
        ledger.transition_attempt(completed["attempt_id"], "READY")
    assert len(ledger.events()) == count


def test_method_completed_phases_match_trial_completeness(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    completed, _ = _complete(ledger, attempt, outcome="accepted")
    trial = ledger.state()["trial_results"][attempt["attempt_id"]]
    assert completed["state"] == "METHOD_COMPLETED"
    assert completed["phases"][trial["completeness"]] == "COMPLETED"
    assert {item["phase"] for item in trial["observations"] if item["role"] == "candidate"} == {trial["completeness"]}


def test_bootstrap_then_standard_same_variant_has_independent_identity_and_budget(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    bootstrap = _reserve(ledger, direction, variant, profile="bootstrap")
    _complete(ledger, bootstrap, outcome="accepted")
    ledger.select_direction(direction)
    ledger.plan_variant(variant, event_id="variant:standard:same-science")
    standard = _reserve(ledger, direction, variant, profile="standard")
    assert standard["attempt_id"] != bootstrap["attempt_id"]
    state = ledger.state()
    assert state["directions"][direction["direction_semantic_hash"]]["budget"] == {"target": 5, "reserved": 1, "consumed": 0}


def test_implementation_repair_keeps_attempt_and_variant_identity(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    ledger.disposition_failure(_failure_evidence(ledger, attempt, "activation_failure"))
    new_implementation = implementation_hash(frozen_patch={"repair": True}, files={"src/router.py": "repaired"}, manifest={"v": 2})
    repaired = ledger.revise_implementation(attempt["attempt_id"], implementation_hash=new_implementation)
    assert repaired["attempt_id"] == attempt["attempt_id"]
    assert repaired["variant_spec_hash"] == attempt["variant_spec_hash"]
    assert repaired["implementation_hash"] != attempt["implementation_hash"]
    assert len(repaired["implementation_revisions"]) == 2


@pytest.mark.parametrize(
    ("failure_class", "expected_action"),
    [("resource_pause", "PAUSE_RESOURCE"), ("activation_failure", "REPAIR_IMPLEMENTATION"), ("integrity_failure", "BLOCK_INTEGRITY")],
)
def test_bootstrap_failures_never_finish_run(tmp_path: Path, failure_class: str, expected_action: str) -> None:
    ledger = ResearchEventLedger(tmp_path / failure_class)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant, profile="bootstrap")
    failed, route = ledger.disposition_failure(_failure_evidence(ledger, attempt, failure_class))
    assert route["next_action"] == expected_action
    assert route["next_action"] != "FINISH_RUN"
    assert failed["method_evaluable"] is False
    assert ledger.state()["directions"][direction["direction_semantic_hash"]]["budget"]["consumed"] == 0


def test_evaluable_trial_rejects_empty_evidence_and_none_completeness(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    completed, _ = _complete(ledger, attempt, outcome="accepted")
    valid = ledger.state()["trial_results"][completed["attempt_id"]]
    mutations = []
    for field, value in [("observed_datasets", []), ("observations", []), ("raw_artifacts", {}), ("completeness", "none")]:
        changed = deepcopy(valid)
        changed[field] = value
        mutations.append(changed)
    for changed in mutations:
        with pytest.raises(ValueError):
            validate_trial_result(changed)


@pytest.mark.parametrize("field", ["variant_id", "variation_coordinates", "intervention", "hypothesis", "null_hypothesis", "alternative_hypothesis", "controlled_variables", "nuisance_variables", "implementation_surface_ids", "expected_metric_signature", "falsification_conditions", "ablation", "resource_budget", "failure_routing", "lineage"])
def test_variant_spec_hash_is_sensitive_to_every_authoritative_field(field: str) -> None:
    direction = _direction()
    variant = _variant(direction, 1)
    changed = deepcopy(variant)
    changed.pop("variant_spec_hash")
    changed.pop("variant_semantic_hash")
    value = changed[field]
    if isinstance(value, str):
        changed[field] = value + "-changed"
    elif isinstance(value, list):
        changed[field] = value + ["changed"]
    else:
        changed[field] = {**value, "changed": True}
    assert variant_spec_hash(changed) != variant["variant_spec_hash"]


@pytest.mark.parametrize("field", ["direction_id", "research_question", "mechanism_invariants", "falsification_conditions", "support_claim_ids", "counter_claim_ids", "implementation_surface_ids", "metric_signature", "benchmark_contract_hash", "variant_space", "exploration_policy", "s2_entry_conditions", "return_to_s1_conditions", "lineage"])
def test_direction_spec_hash_is_sensitive_to_every_authoritative_field(field: str) -> None:
    direction = _direction()
    changed = deepcopy(direction)
    changed.pop("direction_spec_hash")
    changed.pop("direction_semantic_hash")
    value = changed[field]
    if isinstance(value, str):
        changed[field] = value + "-changed"
    elif isinstance(value, list):
        changed[field] = value + ["changed"]
    else:
        changed[field] = {**value, "changed": True}
    assert direction_spec_hash(changed) != direction["direction_spec_hash"]


def test_strict_contract_rejects_missing_extra_and_wrong_version() -> None:
    direction = _direction()
    missing = deepcopy(direction)
    missing.pop("research_question")
    extra = {**direction, "legacy": True}
    wrong = {**direction, "schema_version": "old"}
    for payload in [missing, extra, wrong]:
        with pytest.raises(ValueError):
            validate_contract(payload, "direction_v3.schema.json")


def test_generic_core_state_machine_has_no_c2c_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in ["src/auto_research/domain_contracts.py", "src/auto_research/research_state.py"]:
        assert "c2c" not in (root / rel).read_text(encoding="utf-8").lower()


def test_runtime_source_contains_no_removed_legacy_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    patterns = [
        "literature/ideas.json", "literature/direction_decision.json", "literature/c2c/direction_decision.json",
        "direction_to_legacy_idea", "load_direction_or_legacy_idea", "plan/candidate_ideas.json",
        "plan/next_variant.json", "plan/s2_planner/next_variant.json", "legacy_route_fallback",
    ]
    command = ["rg", "-n", "|".join(pattern.replace(".", r"\.") for pattern in patterns), "src/auto_research", "--glob", "*.py"]
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    assert completed.returncode == 1, completed.stdout

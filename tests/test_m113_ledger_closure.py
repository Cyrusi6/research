from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from copy import deepcopy

import pytest

from auto_research.contract_store import ContractStore
from auto_research.domain_contracts import (
    build_direction_spec,
    build_variant_spec,
    canonical_hash,
)
from auto_research.evidence import (
    content_addressed_evidence_path,
    encode_canonical_evidence,
)
from auto_research.research_state import (
    FAILURE_EVIDENCE_SCHEMA_VERSION,
    RESUME_EVIDENCE_SCHEMA_VERSION,
    IntegrityError,
    ResearchEventLedger,
    _event_hash,
    canonical_json,
    initial_state,
    reduce_event,
)
from support.authoritative_evidence import build_failure_evidence_v4, build_resource_failure_evidence_v4
from support.authoritative_evidence import record_completed_evidence_command


def _direction() -> dict:
    return build_direction_spec(
        {
            "direction_id": "m113-direction",
            "research_question": "Does the intervention improve accuracy?",
            "mechanism_invariants": {
                "causal_hypothesis": "the intervention improves accuracy",
                "target_mediator": "prediction quality",
                "invariants": ["dataset", "evaluator"],
            },
            "falsification_conditions": ["no paired improvement"],
            "support_claim_ids": ["support-1"],
            "counter_claim_ids": ["counter-1"],
            "implementation_surface_ids": ["src/model.py"],
            "metric_signature": {"primary": "accuracy", "direction": "increase"},
            "benchmark_contract_hash": canonical_hash({"benchmark": "fake"}),
            "variant_space": {
                "mutable_axes": ["operation"],
                "immutable_axes": ["dataset", "evaluator"],
                "forbidden_combinations": [{"operation": "forbidden"}],
            },
            "s2_entry_conditions": ["gate passes"],
            "return_to_s1_conditions": ["direction exhausted"],
            "lineage": {
                "s1_run_id": "m113-s1",
                "iteration": 1,
                "input_manifest_hash": canonical_hash({"input": "m113"}),
            },
        }
    )


def _variant(direction: dict) -> dict:
    return build_variant_spec(
        direction,
        {
            "variant_id": "m113-variant",
            "variation_coordinates": {"operation": "calibrated-routing"},
            "intervention": {
                "summary": "calibrated routing",
                "algorithm_operations": ["calibrated-routing"],
                "configuration": {"strength": 1},
            },
            "hypothesis": "routing improves accuracy",
            "null_hypothesis": "routing does not improve accuracy",
            "alternative_hypothesis": "routing improves accuracy",
            "controlled_variables": {"dataset": "fake", "evaluator": "fake-v1"},
            "nuisance_variables": ["noise"],
            "implementation_surface_ids": ["src/model.py"],
            "expected_metric_signature": {"primary": "accuracy", "direction": "increase"},
            "falsification_conditions": ["paired delta is non-positive"],
            "ablation": {"disable": "calibrated-routing"},
            "resource_budget": {"max_wall_seconds": 60, "max_retries": 3},
            "failure_routing": {
                "implementation": "REPAIR_IMPLEMENTATION",
                "method": "PROPOSE_NEXT_VARIANT",
            },
            "lineage": {
                "s2_run_id": "m113-s2",
                "iteration": 1,
                "direction_spec_hash": direction["direction_spec_hash"],
                "feedback_from_attempt_ids": [],
            },
        },
    )


def _trial_spec_legacy() -> dict:
    runtime = {"device": "cpu", "batch_size": 1}
    sample_id = canonical_hash({"sample": "sample-1"})
    sample_dataset = {
        "dataset_id": "fake",
        "source_revision": "synthetic-v1",
        "split": "test",
        "sample_count": 1,
        "ordered_sample_ids": [sample_id],
    }
    sample_dataset["content_hash"] = canonical_hash(
        {
            "dataset_id": sample_dataset["dataset_id"],
            "source_revision": sample_dataset["source_revision"],
            "split": sample_dataset["split"],
            "ordered_sample_ids": sample_dataset["ordered_sample_ids"],
        }
    )
    sample_manifest = {
        "schema_version": "auto_research_sample_manifest_v1",
        "manifest_id": "m113-samples",
        "provenance_mode": "synthetic",
        "datasets": [sample_dataset],
        "artifact_path": "plan/sample_manifest.json",
    }
    sample_manifest["artifact_hash"] = canonical_hash(sample_manifest)
    evaluator_provenance = {
        "schema_version": "auto_research_evaluator_provenance_v1",
        "provenance_mode": "synthetic",
        "evaluator_id": "fake-v1",
        "source_digest": canonical_hash({"source": "fake-evaluator"}),
        "config_hash": canonical_hash({"metric": "accuracy"}),
        "dependency_digest": canonical_hash({"dependencies": []}),
    }
    return {
        "schema_version": "auto_research_trial_spec_v3",
        "protocol": {
            "protocol_id": "m113-full-v1",
            "required_phases": ["full"],
            "terminal_phases": ["full"],
            "proxy_terminal_allowed": False,
            "aggregation": "mean",
        },
        "sample_manifest": sample_manifest,
        "datasets": [
            {
                "dataset_id": "fake",
                "split": "test",
                "sample_count": 1,
                "sample_hash": sample_dataset["content_hash"],
            }
        ],
        "metrics": [
            {
                "metric_id": "accuracy",
                "objective": "maximize",
                "aggregation": "mean",
                "role": "primary",
            }
        ],
        "primary_metric_id": "accuracy",
        "statistical_testing": {
            "method": "none",
            "seeds": [7],
            "require_complete_seed_coverage": True,
        },
        "required_roles": ["baseline", "candidate"],
        "acceptance_constraints": [
            {
                "constraint_id": "primary-delta",
                "kind": "minimum_mean_delta",
                "hard": True,
                "metric_id": "accuracy",
                "threshold": 0.1,
                "objective": "maximize",
            }
        ],
        "execution_contract": {
            "runtime_config": runtime,
            "runtime_config_hash": canonical_hash(runtime),
            "evaluator_provenance": evaluator_provenance,
            "evaluator_hash": canonical_hash(evaluator_provenance),
            "command_contract_hash": canonical_hash({"command": "fake-eval"}),
        },
        "required_artifacts": ["main_results"],
        "evidence_requirements": [
            {
                "requirement_id": "main-results",
                "kind": "main_results",
                "required": True,
                "applicable_phases": ["full"],
                "schema_version": "auto_research_main_results_v2",
            }
        ],
    }



def _trial_spec(project_root: Path | None = None) -> dict:
    from support.authoritative_evidence import build_trial_spec_v5
    return build_trial_spec_v5(_trial_spec_legacy(), project_root=project_root)

def _running_attempt(tmp_path: Path) -> tuple[ResearchEventLedger, dict]:
    from support.authoritative_evidence import start_attempt_phase
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction)
    ledger.select_direction(direction)
    ledger.plan_variant(variant)
    attempt = ledger.reserve_attempt(
        profile="standard",
        direction=direction,
        variant=variant,
        implementation_hash=canonical_hash({"implementation": 1}),
        attempt_kind="full",
        trial_spec=_trial_spec(tmp_path),
    )
    return ledger, start_attempt_phase(ledger, attempt, "full")


def _valid_completion(tmp_path: Path, attempt: dict) -> dict:
    from support.authoritative_evidence import build_quantitative_completion
    return build_quantitative_completion(
        tmp_path,
        attempt,
        role_values={"baseline": 0.0, "candidate": 1.0},
        dataset_id="fake",
        metric_id="accuracy",
        seed=7,
        phase="full",
    )


def _receipt_backed_completion(
    tmp_path: Path,
    ledger: ResearchEventLedger,
    attempt: dict,
) -> dict:
    completion = _valid_completion(tmp_path, attempt)
    record_completed_evidence_command(tmp_path, ledger, attempt, completion)
    return completion

def _forged_trial(tmp_path: Path, attempt: dict) -> dict:
    relative_path = "experiment/results/main_results.json"
    artifact_path = tmp_path / relative_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": "auto_research_main_results_v2",
                "attempt_id": attempt["attempt_id"],
            }
        ),
        encoding="utf-8",
    )
    return {
        "schema_version": "auto_research_trial_result_v4",
        "attempt_id": attempt["attempt_id"],
        "observations": [
            {"role": "baseline", "metric_value": 0.0},
            {"role": "candidate", "metric_value": 1.0},
        ],
        "outcome_classification": "accepted",
        "all_hard_constraints_passed": True,
        "raw_artifacts": {relative_path: hashlib.sha256(artifact_path.read_bytes()).hexdigest()},
    }


def _scoped_artifact(tmp_path: Path, attempt: dict, producer_run_id: str, kind: str, payload: dict) -> str:
    raw = encode_canonical_evidence(payload)
    digest = hashlib.sha256(raw).hexdigest()
    path = (
        tmp_path
        / "experiment"
        / "attempts"
        / attempt["attempt_id"]
        / producer_run_id
        / kind
        / f"{digest}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return digest


def _operation_receipt_binding(tmp_path: Path, attempt: dict, *, label: str) -> dict:
    phase = "full" if attempt["phase_executions"].get("full") else "proxy"
    execution = attempt["phase_executions"][phase]
    receipt_ref = ContractStore(tmp_path).put_bytes(
        encode_canonical_evidence(
            {
                "schema_version": "m113_operation_receipt_fixture_v1",
                "attempt_id": attempt["attempt_id"],
                "lifecycle_generation": attempt["lifecycle_generation"],
                "label": label,
            }
        )
    )
    return {
        "command_id": f"fixture-{label}-{attempt['lifecycle_generation']:04d}",
        "command_hash": canonical_hash(
            {
                "attempt_id": attempt["attempt_id"],
                "lifecycle_generation": attempt["lifecycle_generation"],
                "label": label,
            }
        ),
        "command_plan_hash": execution["command_plan_hash"],
        "receipt_ref": receipt_ref,
        "receipt_hash": receipt_ref["digest"],
    }


def _failure_evidence(
    tmp_path: Path,
    attempt: dict,
    *,
    failure_class: str,
    exit_code: int | None,
    resource_type: str = "gpu_memory",
) -> dict:
    authoritative_exit = int(exit_code or 1) or 1
    if failure_class in {"resource_pause", "oom_retry"}:
        evidence = build_resource_failure_evidence_v4(
            tmp_path,
            attempt,
            failure_class=failure_class,
            suffix=f"m113-{attempt['lifecycle_generation']}",
            resource_type=resource_type,
            resource_id="resource-0",
            required_capacity=10.0,
            observed_capacity=1.0,
            unit="bytes",
            exit_code=authoritative_exit,
        )
    else:
        evidence = build_failure_evidence_v4(
            tmp_path,
            attempt,
            failure_class=failure_class,
            suffix=f"m113-{attempt['lifecycle_generation']}",
            exit_code=authoritative_exit,
        )
    if exit_code != authoritative_exit:
        evidence["exit_code"] = exit_code
        _scoped_artifact(tmp_path, attempt, evidence["producer_run_id"], "failure_evidence", evidence)
    return evidence


def _resume_evidence(tmp_path: Path, ledger: ResearchEventLedger, attempt: dict, *, resource_type: str) -> dict:
    producer_run_id = f"resume-producer-{attempt['lifecycle_generation']}"
    core_receipt = {
        "schema_version": "auto_research_core_resource_probe_receipt_v1",
        "attempt_id": attempt["attempt_id"],
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "resource_type": resource_type,
        "resource_id": "resource-0",
        "required_capacity": 10.0,
        "observed_capacity": 20.0,
        "unit": "bytes",
        "probe_status": "available",
        "observed_at": "2026-07-14T00:01:00Z",
    }
    receipt_ref = ContractStore(tmp_path).put_bytes(encode_canonical_evidence(core_receipt))
    command_id = f"core-resource-probe-{attempt['attempt_id'][:12]}-g{attempt['lifecycle_generation']}"
    receipt_binding = {
        "command_id": command_id,
        "command_hash": canonical_hash({"command_id": command_id, "receipt": receipt_ref["digest"]}),
        "command_plan_hash": canonical_hash({"operation": "resume-resource-probe", "trial_spec_hash": attempt["trial_spec_hash"], "paused_phase": attempt["paused_phase"]}),
        "receipt_ref": receipt_ref,
        "receipt_hash": receipt_ref["digest"],
    }
    probe = {
        "schema_version": "auto_research_resource_probe_evidence_v4", "evidence_kind": "resource_probe",
        "evidence_id": f"resume-probe-{attempt['lifecycle_generation']}", "attempt_id": attempt["attempt_id"],
        "producer_run_id": producer_run_id, "direction_semantic_hash": attempt["direction_semantic_hash"],
        "direction_spec_hash": attempt["direction_spec_hash"], "variant_semantic_hash": attempt["variant_semantic_hash"],
        "variant_spec_hash": attempt["variant_spec_hash"], "trial_spec_hash": attempt["trial_spec_hash"],
        "protocol_hash": attempt["protocol_hash"], "sample_manifest_hash": attempt["sample_manifest_hash"],
        "evaluator_hash": attempt["evaluator_hash"], "resource_type": resource_type,
        "resource_id": "resource-0", "required_capacity": 10.0, "observed_capacity": 20.0, "unit": "bytes",
        "probe_status": "available", "observed_at": "2026-07-14T00:01:00Z",
        "lifecycle_generation": attempt["lifecycle_generation"], "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"], "phase": "resume",
        "phase_execution_id": f"phase-resume-{attempt['lifecycle_generation']:04d}",
        "phase_start_event_id": f"event:resume:{attempt['lifecycle_generation']}",
        **receipt_binding,
    }
    probe_hash = _scoped_artifact(tmp_path, attempt, producer_run_id, "resource_probe", probe)
    pause_event = next(event for event in reversed(ledger.events()) if event["event_type"] == "AttemptDispositioned")
    pause_evidence = pause_event["payload"]["failure_evidence"]
    evidence = {
        "schema_version": RESUME_EVIDENCE_SCHEMA_VERSION,
        "evidence_kind": "resume_evidence", "evidence_id": f"resume-evidence-{attempt['lifecycle_generation']}",
        "attempt_id": attempt["attempt_id"],
        "producer_run_id": producer_run_id, "direction_semantic_hash": attempt["direction_semantic_hash"],
        "direction_spec_hash": attempt["direction_spec_hash"], "variant_semantic_hash": attempt["variant_semantic_hash"],
        "variant_spec_hash": attempt["variant_spec_hash"], "trial_spec_hash": attempt["trial_spec_hash"],
        "protocol_hash": attempt["protocol_hash"], "sample_manifest_hash": attempt["sample_manifest_hash"],
        "evaluator_hash": attempt["evaluator_hash"], "cross_references": {"resource_probe_hash": probe_hash},
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "phase": "resume",
        "phase_execution_id": probe["phase_execution_id"],
        "phase_start_event_id": probe["phase_start_event_id"],
        "pause_event_id": pause_event["event_id"], "pause_evidence_hash": hashlib.sha256(encode_canonical_evidence(pause_evidence)).hexdigest(),
        "pause_phase": pause_evidence["phase"],
        "pause_phase_execution_id": pause_evidence["phase_execution_id"],
        "pause_producer_run_id": pause_evidence["producer_run_id"],
        "resource_type": resource_type, "resource_id": "resource-0", "required_capacity": 10.0,
        "observed_capacity": 20.0, "unit": "bytes", "probe_status": "available",
        "observed_at": "2026-07-14T00:01:00Z",
        **receipt_binding,
    }
    _scoped_artifact(tmp_path, attempt, producer_run_id, "resume_evidence", evidence)
    return evidence


def test_identity_only_artifact_cannot_finalize_with_forged_observations(tmp_path: Path) -> None:
    ledger, attempt = _running_attempt(tmp_path)
    valid = _receipt_backed_completion(tmp_path, ledger, attempt)
    canonical = ledger.validate_trial_precommit(valid)
    assert canonical["outcome_classification"] == "accepted"
    before = ledger.state()
    forged = _forged_trial(tmp_path, attempt)
    assert forged["outcome_classification"] == "accepted"

    with pytest.raises(IntegrityError, match="evidence|measurement|row|[Cc]ompletion"):
        ledger.complete_attempt(forged)

    after = ledger.state()
    assert after["last_sequence"] == before["last_sequence"]
    assert after["trial_results"] == before["trial_results"] == {}
    assert after["last_route_outcome"] == before["last_route_outcome"]
    budget = after["directions"][attempt["direction_semantic_hash"]]["budget"]
    assert budget == {"target": 5, "reserved": 1, "consumed": 0}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda attempt: attempt.__setitem__("state", "ABANDONED"),
        lambda attempt: attempt.__setitem__("terminal_outcome", "accepted"),
        lambda attempt: attempt["phases"].__setitem__("full", "COMPLETED"),
        lambda attempt: attempt["artifact_hashes"].__setitem__("forged.json", "0" * 64),
    ],
)
def test_attempt_reserved_reducer_rejects_noncanonical_initial_record(
    tmp_path: Path, mutation
) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction)
    ledger.select_direction(direction)
    ledger.plan_variant(variant)
    ledger.reserve_attempt(
        profile="standard",
        direction=direction,
        variant=variant,
        implementation_hash=canonical_hash({"implementation": 1}),
        attempt_kind="full",
        trial_spec=_trial_spec(tmp_path),
    )
    events = ledger.events()
    state = initial_state(tmp_path.name)
    state = reduce_event(state, events[0])
    state = reduce_event(state, events[1])
    forged = deepcopy(events[2])
    mutation(forged["payload"]["attempt"])

    with pytest.raises(IntegrityError, match="AttemptReserved|canonical|initial"):
        reduce_event(state, forged)


def test_resource_pause_with_success_exit_code_is_zero_write_rejected(tmp_path: Path) -> None:
    ledger, attempt = _running_attempt(tmp_path)
    evidence = _failure_evidence(
        tmp_path, attempt, failure_class="resource_pause", exit_code=0
    )
    before = ledger.state()

    with pytest.raises(IntegrityError, match="exit|resource|failure"):
        ledger.disposition_failure(evidence)

    after = ledger.state()
    assert after["last_sequence"] == before["last_sequence"]
    assert after["attempts"][attempt["attempt_id"]] == before["attempts"][attempt["attempt_id"]]


def test_resume_resource_must_match_pause_resource(tmp_path: Path) -> None:
    ledger, attempt = _running_attempt(tmp_path)
    pause = _failure_evidence(
        tmp_path, attempt, failure_class="resource_pause", exit_code=137, resource_type="gpu_memory"
    )
    paused, _ = ledger.disposition_failure(pause)
    before = ledger.state()

    with pytest.raises(IntegrityError, match="resource|pause"):
        ledger.resume_attempt(_resume_evidence(tmp_path, ledger, paused, resource_type="network"))

    after = ledger.state()
    assert after["last_sequence"] == before["last_sequence"]
    assert after["attempts"][attempt["attempt_id"]]["state"] == "RESOURCE_PAUSED"


def test_revision_cycle_back_to_prior_hash_is_not_implicit_late_replay(tmp_path: Path) -> None:
    ledger, attempt = _running_attempt(tmp_path)
    implementation_b = canonical_hash({"implementation": "B"})
    implementation_c = canonical_hash({"implementation": "C"})

    failed, _ = ledger.disposition_failure(
        _failure_evidence(tmp_path, attempt, failure_class="activation_failure", exit_code=1)
    )
    revision_b = ledger.revise_implementation(
        failed["attempt_id"], implementation_hash=implementation_b
    )
    from support.authoritative_evidence import start_attempt_phase
    running_b = start_attempt_phase(ledger, revision_b, "full")
    failed_b, _ = ledger.disposition_failure(
        _failure_evidence(tmp_path, running_b, failure_class="activation_failure", exit_code=1)
    )
    revision_c = ledger.revise_implementation(
        failed_b["attempt_id"], implementation_hash=implementation_c
    )
    running_c = start_attempt_phase(ledger, revision_c, "full")
    failed_c, _ = ledger.disposition_failure(
        _failure_evidence(tmp_path, running_c, failure_class="activation_failure", exit_code=1)
    )
    before_sequence = ledger.state()["last_sequence"]

    current_b = ledger.revise_implementation(
        failed_c["attempt_id"], implementation_hash=implementation_b
    )

    assert ledger.state()["last_sequence"] == before_sequence + 1
    assert current_b["implementation_hash"] == implementation_b
    assert current_b["lifecycle_generation"] == 3


def test_completion_public_validation_and_reducer_use_same_canonical_trial(tmp_path: Path) -> None:
    ledger, attempt = _running_attempt(tmp_path)
    completion = _receipt_backed_completion(tmp_path, ledger, attempt)
    canonical = ledger.validate_trial_precommit(completion)

    completed, route = ledger.complete_attempt(completion)
    event = ledger.events()[-1]

    assert event["event_type"] == "AttemptFinalized"
    assert event["payload"]["trial_result"] == canonical
    assert ledger.rebuild()["trial_results"][attempt["attempt_id"]] == canonical
    assert completed["state"] == "METHOD_COMPLETED"
    assert route["source"]["event_id"] == event["event_id"]


def test_orphan_evidence_before_db_can_retry_without_duplicate_commit(tmp_path: Path) -> None:
    ledger, attempt = _running_attempt(tmp_path)
    completion = _receipt_backed_completion(tmp_path, ledger, attempt)
    canonical = ledger.validate_trial_precommit(completion)
    invalid = deepcopy(completion)
    invalid["diagnostic_trial_result"] = {**canonical, "outcome_classification": "rejected"}
    before = len(ledger.events())
    artifact = tmp_path / completion["entries"][0]["relative_path"]
    assert artifact.is_file()

    with pytest.raises(IntegrityError, match="CompletionEvidence v3|diagnostic TrialResult"):
        ledger.complete_attempt(invalid)

    assert artifact.is_file()
    assert len(ledger.events()) == before
    ledger.complete_attempt(completion, event_id="finalize:orphan-retry")
    assert len(ledger.events()) == before + 1
    ledger.complete_attempt(completion, event_id="finalize:orphan-retry")
    assert len(ledger.events()) == before + 1


def test_db_commit_before_projection_crash_recovers_one_result_and_route(tmp_path: Path) -> None:
    ledger, attempt = _running_attempt(tmp_path)
    completion = _receipt_backed_completion(tmp_path, ledger, attempt)

    def crash() -> None:
        raise RuntimeError("after-db-before-projection")

    ledger.after_commit_hook = crash
    with pytest.raises(RuntimeError, match="after-db-before-projection"):
        ledger.complete_attempt(completion, event_id="finalize:crash-recovery")

    resumed = ResearchEventLedger(tmp_path)
    state = resumed.rebuild()
    assert len(state["trial_results"]) == 1
    assert state["directions"][attempt["direction_semantic_hash"]]["budget"] == {
        "target": 5,
        "reserved": 0,
        "consumed": 1,
    }
    sequence = state["last_sequence"]
    resumed.complete_attempt(completion, event_id="finalize:crash-recovery")
    assert resumed.state()["last_sequence"] == sequence


def test_missing_trial_spec_projection_rejects_completion_with_zero_write(tmp_path: Path) -> None:
    ledger, attempt = _running_attempt(tmp_path)
    completion = _receipt_backed_completion(tmp_path, ledger, attempt)
    ledger.validate_trial_precommit(completion)
    before = len(ledger.events())
    (tmp_path / "plan" / "attempts" / attempt["attempt_id"] / "trial_spec" / f"{attempt['trial_spec_hash']}.json").unlink()

    with pytest.raises(IntegrityError, match="TrialSpec projection"):
        ledger.complete_attempt(completion)

    assert len(ledger.events()) == before
    assert not (tmp_path / "experiment" / "results" / "trial_result.json").exists()


def test_attempt_reserved_sqlite_tamper_is_rejected_during_rebuild(tmp_path: Path) -> None:
    ledger, _attempt = _running_attempt(tmp_path)
    reserved = next(event for event in ledger.events() if event["event_type"] == "AttemptReserved")
    forged = deepcopy(reserved)
    forged["payload"]["attempt"]["state"] = "ABANDONED"
    forged["event_hash"] = _event_hash(forged)
    with sqlite3.connect(ledger.db_path) as connection:
        connection.execute(
            "UPDATE events SET payload_json = ?, event_hash = ? WHERE sequence = ?",
            (canonical_json(forged["payload"]), forged["event_hash"], forged["sequence"]),
        )
        following = connection.execute(
            "SELECT sequence, payload_json, event_id, event_type, created_at, schema_version FROM events WHERE sequence > ? ORDER BY sequence",
            (forged["sequence"],),
        ).fetchall()
        previous_hash = forged["event_hash"]
        for sequence, payload_json, event_id, event_type, created_at, schema_version in following:
            event = {
                "schema_version": schema_version,
                "event_id": event_id,
                "sequence": sequence,
                "event_type": event_type,
                "previous_event_hash": previous_hash,
                "created_at": created_at,
                "payload": json.loads(payload_json),
            }
            event_hash = _event_hash(event)
            connection.execute(
                "UPDATE events SET previous_event_hash = ?, event_hash = ? WHERE sequence = ?",
                (previous_hash, event_hash, sequence),
            )
            previous_hash = event_hash

    with pytest.raises(IntegrityError, match="canonical initial Attempt"):
        ledger.rebuild()

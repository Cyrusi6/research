from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from pathlib import Path

import pytest

from auto_research.domain_contracts import canonical_hash
from auto_research.failure_validation import canonical_evidence_bytes, evidence_bytes_hash
from auto_research.research_state import IntegrityError, ResearchEventLedger, _event_hash, canonical_json
from auto_research.agents.plan import _trial_spec_from_plan
from test_m113_ledger_closure import _direction, _variant
from auto_research.agents.experiment import _c2c_strict_evidence_inventory, _stage_evidence_inventory


def _stage_operation_bytes(
    project_root: Path,
    attempt: dict,
    producer_run_id: str,
    kind: str,
    payload: dict,
) -> str:
    raw = canonical_evidence_bytes(payload)
    digest = evidence_bytes_hash(raw)
    path = project_root / "experiment" / "attempts" / attempt["attempt_id"] / producer_run_id / kind / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return digest


def _full_running_proxy_attempt(tmp_path: Path) -> tuple[ResearchEventLedger, dict]:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction)
    variant_path = tmp_path / "plan" / "variant.json"
    variant_path.parent.mkdir(parents=True, exist_ok=True)
    variant_path.write_text(json.dumps(variant, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    plan = {
        "datasets": [{"name": "fake", "split": "test", "sample_count": 1}],
        "metrics": [{"name": "accuracy", "primary": True, "higher_is_better": True}],
        "statistical_testing": {"seeds": [7]},
        "acceptance_criteria": {"minimum_mean_delta": 0.1, "maximum_dataset_regression": 0.0},
        "ablation_matrix": [],
        "execution": {
            "mode": "simulate",
            "collector": "c2c_small_loop",
            "commands": [],
            "evaluator_id": "synthetic-evaluator",
        },
    }
    trial_spec = _trial_spec_from_plan(plan, variant, project_root=tmp_path)
    ledger.select_direction(direction)
    ledger.plan_variant(variant)
    attempt = ledger.reserve_attempt(
        profile="standard",
        direction=direction,
        variant=variant,
        implementation_hash=canonical_hash({"implementation": "resource-resume"}),
        attempt_kind="proxy_full",
        trial_spec=trial_spec,
    )
    attempt = ledger.start_proxy_phase(
        attempt["attempt_id"],
        phase_execution_id="phase-proxy-resource-0001",
        producer_run_id="producer-proxy-resource-0001",
    )
    comparison = {
        "metrics": {"mean": 1.0, "datasets": {"fake": 1.0}},
        "proxy_screen": {
            "metrics": {"mean": 1.0, "datasets": {"fake": 1.0}},
            "baseline_metrics": {"mean": 0.0, "datasets": {"fake": 0.0}},
        },
    }
    baseline = {"mean": 0.0, "datasets": {"fake": 0.0}}
    inventory = _c2c_strict_evidence_inventory(
        project_root=tmp_path,
        attempt=attempt,
        trial_spec=trial_spec,
        comparison_candidate=comparison,
        baseline=baseline,
        simulate=False,
    )
    inventory = [
        item for item in inventory
        if item.get("kind") not in {"effective_proxy_policy", "proxy_calibration_policy"}
    ]
    completion = _stage_evidence_inventory(
        project_root=tmp_path,
        attempt=attempt,
        trial_spec=trial_spec,
        inventory=inventory,
    )
    proxy_completed, route = ledger.commit_proxy_evidence(completion)
    assert route["next_action"] == "RUN_FULL"
    return ledger, ledger.start_full_phase(
        proxy_completed["attempt_id"],
        phase_execution_id="phase-full-resource-0001",
        producer_run_id="producer-full-resource-0001",
    )


def _resource_pause(tmp_path: Path, attempt: dict) -> tuple[dict, dict, str]:
    execution = attempt["phase_executions"]["full"]
    identity = {
        "attempt_id": attempt["attempt_id"],
        "producer_run_id": execution["producer_run_id"],
        "direction_semantic_hash": attempt["direction_semantic_hash"],
        "direction_spec_hash": attempt["direction_spec_hash"],
        "variant_semantic_hash": attempt["variant_semantic_hash"],
        "variant_spec_hash": attempt["variant_spec_hash"],
        "trial_spec_hash": attempt["trial_spec_hash"],
        "protocol_hash": attempt["protocol_hash"],
        "sample_manifest_hash": attempt["sample_manifest_hash"],
        "evaluator_hash": attempt["evaluator_hash"],
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "phase": "full",
        "phase_execution_id": execution["phase_execution_id"],
        "phase_start_event_id": execution["phase_start_event_id"],
    }
    probe = {
        "schema_version": "auto_research_resource_probe_evidence_v3",
        "evidence_kind": "resource_probe",
        "evidence_id": "resource-probe-full-0001",
        **identity,
        "resource_type": "gpu_memory",
        "resource_id": "gpu:0",
        "required_capacity": 10.0,
        "observed_capacity": 4.0,
        "unit": "bytes",
        "probe_status": "insufficient",
        "observed_at": "2026-07-15T01:00:00Z",
    }
    probe_hash = _stage_operation_bytes(tmp_path, attempt, identity["producer_run_id"], "resource_probe", probe)
    failure = {
        "schema_version": "auto_research_failure_evidence_v4",
        "evidence_kind": "failure_evidence",
        "evidence_id": "failure-resource-full-0001",
        **identity,
        "cross_references": {"resource_probe_hash": probe_hash},
        "source_state": "FULL_RUNNING",
        "source_phase": "full",
        "failure_class": "resource_pause",
        "command_status": "resource_paused",
        "exit_code": 137,
        "reason": "gpu memory insufficient",
        "observed_at": "2026-07-15T01:00:01Z",
        "log_hash": probe_hash,
    }
    failure_hash = _stage_operation_bytes(tmp_path, attempt, identity["producer_run_id"], "failure_evidence", failure)
    return failure, probe, failure_hash


def _resume(tmp_path: Path, attempt: dict, pause_event: dict, pause_failure: dict, pause_hash: str) -> dict:
    producer_run_id = "producer-resume-resource-0001"
    identity = {
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
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "phase": "resume",
        "phase_execution_id": "phase-resume-resource-0001",
        "phase_start_event_id": "event:resume:resource:0001",
    }
    probe = {
        "schema_version": "auto_research_resource_probe_evidence_v3",
        "evidence_kind": "resource_probe",
        "evidence_id": "resource-probe-resume-0001",
        **identity,
        "resource_type": "gpu_memory",
        "resource_id": "gpu:0",
        "required_capacity": 10.0,
        "observed_capacity": 20.0,
        "unit": "bytes",
        "probe_status": "available",
        "observed_at": "2026-07-15T01:01:00Z",
    }
    probe_hash = _stage_operation_bytes(tmp_path, attempt, producer_run_id, "resource_probe", probe)
    resume = {
        "schema_version": "auto_research_resume_evidence_v4",
        "evidence_kind": "resume_evidence",
        "evidence_id": "resume-resource-full-0001",
        **identity,
        "cross_references": {"resource_probe_hash": probe_hash},
        "pause_event_id": pause_event["event_id"],
        "pause_evidence_hash": pause_hash,
        "pause_phase": "full",
        "pause_phase_execution_id": pause_failure["phase_execution_id"],
        "pause_producer_run_id": pause_failure["producer_run_id"],
        "resource_type": "gpu_memory",
        "resource_id": "gpu:0",
        "required_capacity": 10.0,
        "observed_capacity": 20.0,
        "unit": "bytes",
        "probe_status": "available",
        "observed_at": "2026-07-15T01:01:00Z",
    }
    _stage_operation_bytes(tmp_path, attempt, producer_run_id, "resume_evidence", resume)
    return resume


def _rehash_from(ledger: ResearchEventLedger, sequence: int, replacement_payload: dict) -> None:
    with sqlite3.connect(ledger.db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM events ORDER BY sequence").fetchall()
        previous_hash = "0" * 64
        for row in rows:
            current_sequence = row["sequence"]
            event = {
                "schema_version": row["schema_version"],
                "event_id": row["event_id"],
                "sequence": current_sequence,
                "event_type": row["event_type"],
                "previous_event_hash": row["previous_event_hash"],
                "event_hash": row["event_hash"],
                "created_at": row["created_at"],
                "payload": json.loads(row["payload_json"]),
            }
            if current_sequence == sequence:
                event["payload"] = replacement_payload
            event["previous_event_hash"] = previous_hash
            event["event_hash"] = _event_hash(event)
            connection.execute(
                "UPDATE events SET payload_json=?, previous_event_hash=?, event_hash=? WHERE sequence=?",
                (canonical_json(event["payload"]), previous_hash, event["event_hash"], current_sequence),
            )
            previous_hash = event["event_hash"]
        connection.commit()


def test_full_resource_resume_retains_proxy_authorization_and_restarts_only_full(tmp_path: Path) -> None:
    ledger, running = _full_running_proxy_attempt(tmp_path)
    committed_proxy = deepcopy(running["committed_proxy_outcome"])
    failure, _, failure_hash = _resource_pause(tmp_path, running)
    paused, route = ledger.disposition_failure(failure)
    assert route["next_action"] == "PAUSE_RESOURCE"
    assert paused["state"] == "RESOURCE_PAUSED"
    assert paused["phases"] == {"proxy": "COMPLETED", "full": "RUNNING"}

    pause_event = next(event for event in reversed(ledger.events()) if event["event_type"] == "AttemptDispositioned")
    resumed = ledger.resume_attempt(_resume(tmp_path, paused, pause_event, failure, failure_hash))

    assert resumed["lifecycle_generation"] == running["lifecycle_generation"] + 1
    assert resumed["state"] == "PROXY_COMPLETED"
    assert resumed["phases"] == {"proxy": "COMPLETED", "full": "PENDING"}
    assert resumed["committed_proxy_outcome"] == committed_proxy
    restarted = ledger.start_full_phase(
        resumed["attempt_id"],
        phase_execution_id="phase-full-resource-0002",
        producer_run_id="producer-full-resource-0002",
    )
    assert restarted["state"] == "FULL_RUNNING"
    assert restarted["phases"] == {"proxy": "COMPLETED", "full": "RUNNING"}


def test_rebuild_rejects_rehashed_failure_semantic_mutation(tmp_path: Path) -> None:
    ledger, running = _full_running_proxy_attempt(tmp_path)
    failure, _, _ = _resource_pause(tmp_path, running)
    ledger.disposition_failure(failure)
    event = next(event for event in ledger.events() if event["event_type"] == "AttemptDispositioned")
    attacked = deepcopy(event["payload"])
    attacked["failure_evidence"]["exit_code"] = 0
    forged = attacked["failure_evidence"]
    _stage_operation_bytes(
        tmp_path,
        running,
        forged["producer_run_id"],
        "failure_evidence",
        forged,
    )
    _rehash_from(ledger, event["sequence"], attacked)

    with pytest.raises(IntegrityError, match="FailureEvidence|resource|schema|raw-byte"):
        ledger.rebuild()


def test_rebuild_rejects_failure_and_resume_raw_byte_drift(tmp_path: Path) -> None:
    ledger, running = _full_running_proxy_attempt(tmp_path)
    failure, _, failure_hash = _resource_pause(tmp_path, running)
    paused, _ = ledger.disposition_failure(failure)
    pause_event = next(event for event in reversed(ledger.events()) if event["event_type"] == "AttemptDispositioned")
    resume = _resume(tmp_path, paused, pause_event, failure, failure_hash)
    ledger.resume_attempt(resume)

    resume_hash = evidence_bytes_hash(canonical_evidence_bytes(resume))
    resume_path = tmp_path / "experiment" / "attempts" / paused["attempt_id"] / resume["producer_run_id"] / "resume_evidence" / f"{resume_hash}.json"
    resume_path.write_bytes(canonical_evidence_bytes({**resume, "observed_capacity": 21.0}))

    with pytest.raises(IntegrityError, match="resume_evidence artifact hash drift|ResumeEvidence"):
        ledger.rebuild()

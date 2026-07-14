from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import auto_research.orchestrator as orchestrator_module
from auto_research.domain_contracts import EXECUTION_OBSERVATION_SCHEMA_VERSION, classify_trial_result
from auto_research.orchestrator import Orchestrator
from auto_research.registry import load_registry, save_registry
from auto_research.research_state import IntegrityError, ResearchEventLedger
from auto_research.utils import write_json
from test_authoritative_state_machine import _direction, _initialize, _reserve, _trial_spec, _variant
from test_pipeline import _test_config


def _trial(
    ledger: ResearchEventLedger,
    attempt: dict,
    *,
    baseline: float = 0.5,
    candidate: float = 0.7,
) -> dict:
    artifact = ledger.project_root / "experiment" / "raw" / f"{attempt['attempt_id']}.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('{"verified": true}\n', encoding="utf-8")
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    phase = "proxy" if attempt["attempt_kind"] == "bootstrap_proxy" else "full"
    observations = [
        {
            "schema_version": EXECUTION_OBSERVATION_SCHEMA_VERSION,
            "phase": phase,
            "role": role,
            "command_status": "completed",
            "dataset_id": "fake",
            "metric_id": "accuracy",
            "metric_value": value,
            "sample_manifest_hash": attempt["sample_manifest_hash"],
            "evaluator_hash": attempt["evaluator_hash"],
            "seed": 1,
            "raw_artifact_hash": artifact_hash,
        }
        for role, value in (("baseline", baseline), ("candidate", candidate))
    ]
    return classify_trial_result(
        attempt=attempt,
        trial_spec=_trial_spec(attempt),
        observations=observations,
        raw_artifacts={str(artifact.relative_to(ledger.project_root)): artifact_hash},
    )


def _prepared_attempt(tmp_path: Path, *, bootstrap: bool = False) -> tuple[ResearchEventLedger, dict, dict, dict]:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant, profile="bootstrap" if bootstrap else "standard")
    phase = "proxy" if bootstrap else "full"
    ledger.transition_attempt(attempt["attempt_id"], "PROXY_RUNNING" if bootstrap else "FULL_RUNNING", phase=phase, phase_state="RUNNING")
    return ledger, direction, variant, ledger.state()["attempts"][attempt["attempt_id"]]


def _assert_precommit_rejected_without_authoritative_write(
    ledger: ResearchEventLedger,
    direction: dict,
    trial: dict,
) -> None:
    before = ledger.state()
    before_events = len(ledger.events())
    before_trial_path = ledger.project_root / "experiment" / "results" / "trial_result.json"
    before_trial_bytes = before_trial_path.read_bytes() if before_trial_path.exists() else None

    with pytest.raises((IntegrityError, ValueError)):
        ledger.complete_attempt(trial)

    after = ledger.state()
    assert len(ledger.events()) == before_events
    assert after["directions"][direction["direction_semantic_hash"]]["budget"] == before["directions"][direction["direction_semantic_hash"]]["budget"]
    assert after["method_tried_history"] == before["method_tried_history"]
    assert after["last_route_outcome"] == before["last_route_outcome"]
    assert after["trial_results"] == before["trial_results"]
    assert (before_trial_path.read_bytes() if before_trial_path.exists() else None) == before_trial_bytes
    assert all(event["event_type"] != "AttemptFinalized" for event in ledger.events())


@pytest.mark.parametrize(
    "invalid_case",
    [
        "proxy_contract",
        "artifact_hash",
        "dataset_coverage",
        "phase_mismatch",
        "identity_mismatch",
    ],
)
def test_s3_precommit_rejects_invalid_trial_without_finalization_or_budget_change(
    tmp_path: Path,
    invalid_case: str,
) -> None:
    ledger, direction, _, attempt = _prepared_attempt(tmp_path)
    trial = _trial(ledger, attempt)

    if invalid_case == "proxy_contract":
        write_json(
            tmp_path / "experiment" / "results" / "c2c_proxy_decision_report.json",
            {"schema_version": "c2c_proxy_decision_report_v1", "decision": "forged"},
        )
    elif invalid_case == "artifact_hash":
        trial["raw_artifacts"][next(iter(trial["raw_artifacts"]))] = "0" * 64
    elif invalid_case == "dataset_coverage":
        for observation in trial["observations"]:
            observation["dataset_id"] = "unregistered-dataset"
    elif invalid_case == "phase_mismatch":
        for observation in trial["observations"]:
            observation["phase"] = "proxy"
    elif invalid_case == "identity_mismatch":
        trial["attempt_input_hash"] = "f" * 64

    _assert_precommit_rejected_without_authoritative_write(ledger, direction, trial)


def test_bootstrap_precommit_requires_verified_completion_before_finish_run(tmp_path: Path) -> None:
    ledger, direction, _, attempt = _prepared_attempt(tmp_path, bootstrap=True)
    trial = _trial(ledger, attempt)

    _assert_precommit_rejected_without_authoritative_write(ledger, direction, trial)


class _RouteObserved(RuntimeError):
    pass


def _run_s3_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    next_action: str,
) -> dict:
    config = _test_config(tmp_path, simulate=True)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)
    project_id = Orchestrator().init_project("route authority", project_id=f"route-{next_action.lower()}", simulate=True)
    project_root = tmp_path / project_id
    registry_path = project_root / "meta" / "registry.yaml"
    registry = load_registry(registry_path)
    registry["current_stage"] = "S3_experiment"
    registry["status"] = "running"
    save_registry(registry_path, registry)

    route = {"next_action": next_action, "source": {"attempt_id": "attempt-route"}}
    monkeypatch.setattr(
        orchestrator_module.ExperimentAgent,
        "run",
        lambda self: {
            "status": "blocked",
            "blocked_reason": "diagnostic status must not override reducer route",
            "attempt": {"attempt_id": "attempt-route"},
            "route_outcome": route,
            "artifacts": [],
        },
    )
    monkeypatch.setattr(
        orchestrator_module,
        "gate_s3",
        lambda *args, **kwargs: SimpleNamespace(
            legacy_tuple=lambda: (True, ""),
            to_dict=lambda: {"schema_version": "stage_gate_v1", "stage": "S3_experiment", "status": "PASS", "checks": []},
        ),
    )
    monkeypatch.setattr(
        orchestrator_module.StageContractManager,
        "required_input_status",
        lambda self, *args, **kwargs: {"missing_inputs": [], "required_inputs": []},
    )
    if next_action == "REPAIR_IMPLEMENTATION":
        def observe_invalidation(registry_payload, stage_key, *, invalidated_by):
            assert stage_key == "S2_plan"
            assert invalidated_by == "route_outcome:REPAIR_IMPLEMENTATION"
            raise _RouteObserved

        monkeypatch.setattr(orchestrator_module, "invalidate_from", observe_invalidation)
        with pytest.raises(_RouteObserved):
            Orchestrator().start(project_id)
        return {"status": "repair_observed"}
    return Orchestrator().start(project_id)


def test_orchestrator_prefers_repair_route_over_blocked_result_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert _run_s3_route(monkeypatch, tmp_path, next_action="REPAIR_IMPLEMENTATION")["status"] == "repair_observed"


def test_orchestrator_prefers_pause_route_over_blocked_result_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = _run_s3_route(monkeypatch, tmp_path, next_action="PAUSE_RESOURCE")
    assert result["status"] == "retryable_paused"
    assert result["attempt_id"] == "attempt-route"


def test_orchestrator_prefers_integrity_route_over_blocked_result_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = _run_s3_route(monkeypatch, tmp_path, next_action="BLOCK_INTEGRITY")
    assert result["status"] == "blocked"
    assert result["reason"] == "Integrity block recorded by the attempt reducer"


def test_direction_aggregate_selects_best_accepted_delta_not_highest_absolute_candidate(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    expected_best_attempt_id = None
    rejected_high_candidate_attempt_id = None

    measurements = [
        (0.50, 0.70),
        (1.00, 0.95),
        (0.60, 0.68),
        (0.55, 0.57),
        (0.40, 0.46),
    ]
    for index, (baseline, candidate) in enumerate(measurements, start=1):
        variant = _variant(direction, index)
        _initialize(ledger, direction, variant)
        attempt = _reserve(ledger, direction, variant)
        ledger.transition_attempt(attempt["attempt_id"], "FULL_RUNNING", phase="full", phase_state="RUNNING")
        trial = _trial(ledger, ledger.state()["attempts"][attempt["attempt_id"]], baseline=baseline, candidate=candidate)
        ledger.complete_attempt(trial)
        if index == 1:
            expected_best_attempt_id = attempt["attempt_id"]
        if index == 2:
            rejected_high_candidate_attempt_id = attempt["attempt_id"]

    aggregate = ledger.state()["latest_direction_aggregate"]
    assert next(item for item in aggregate["outcomes"] if item["attempt_id"] == rejected_high_candidate_attempt_id)["outcome"] == "rejected"
    assert aggregate["selection"]["best_attempt_id"] == expected_best_attempt_id
    assert aggregate["selection"]["best_attempt_id"] != rejected_high_candidate_attempt_id


from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from auto_research.domain_contracts import EVIDENCE_MANIFEST_SCHEMA_VERSION, EXECUTION_OBSERVATION_SCHEMA_VERSION, classify_trial_result
from auto_research.research_state import IntegrityError, ResearchEventLedger
from auto_research.utils import write_json
from test_authoritative_state_machine import _direction, _initialize, _reserve, _trial_spec, _variant


def _trial(
    ledger: ResearchEventLedger,
    attempt: dict,
    *,
    baseline: float = 0.5,
    candidate: float = 0.7,
) -> dict:
    artifact = ledger.project_root / "experiment" / "raw" / f"{attempt['attempt_id']}.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    write_json(artifact, {"schema_version": "auto_research_main_results_v1", "attempt_id": attempt["attempt_id"]})
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    phase = "proxy" if attempt["attempt_kind"] == "bootstrap_proxy" else "full"
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
        for role, value in (("baseline", baseline), ("candidate", candidate))
    ]
    manifest = {
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
        evidence_manifest=manifest,
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
        "evidence_cross_reference",
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

    if invalid_case == "evidence_cross_reference":
        trial["evidence_manifest"]["entries"][0]["trial_spec_hash"] = "0" * 64
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
    trial["evidence_manifest"]["entries"] = []
    trial["raw_artifacts"] = {}
    _assert_precommit_rejected_without_authoritative_write(ledger, direction, trial)


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

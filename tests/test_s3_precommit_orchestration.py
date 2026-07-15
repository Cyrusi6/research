from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from auto_research.evidence import content_addressed_evidence_path, encode_canonical_evidence
from auto_research.research_state import IntegrityError, ResearchEventLedger
from test_authoritative_state_machine import (
    _complete,
    _completion_evidence,
    _direction,
    _initialize,
    _reserve,
    _variant,
)
from support.authoritative_evidence import record_completed_evidence_command, start_attempt_phase


def _prepared_attempt(tmp_path: Path, *, bootstrap: bool = False) -> tuple[ResearchEventLedger, dict, dict, dict]:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant, profile="bootstrap" if bootstrap else "standard")
    phase = "proxy" if bootstrap else "full"
    attempt = start_attempt_phase(ledger, attempt, phase)
    return ledger, direction, variant, attempt


def _assert_precommit_rejected_without_authoritative_write(
    ledger: ResearchEventLedger,
    direction: dict,
    completion: dict,
) -> None:
    before = ledger.state()
    before_events = len(ledger.events())
    before_trial_path = ledger.project_root / "experiment" / "results" / "trial_result.json"
    before_trial_bytes = before_trial_path.read_bytes() if before_trial_path.exists() else None

    with pytest.raises((IntegrityError, ValueError)):
        ledger.complete_attempt(completion)

    after = ledger.state()
    assert len(ledger.events()) == before_events
    assert after["directions"][direction["direction_semantic_hash"]]["budget"] == before["directions"][direction["direction_semantic_hash"]]["budget"]
    assert after["method_tried_history"] == before["method_tried_history"]
    assert after["last_route_outcome"] == before["last_route_outcome"]
    assert after["trial_results"] == before["trial_results"]
    assert (before_trial_path.read_bytes() if before_trial_path.exists() else None) == before_trial_bytes
    assert all(event["event_type"] != "AttemptFinalized" for event in ledger.events())


def _rewrite_payload(ledger: ResearchEventLedger, completion: dict, mutate) -> None:
    entry = completion["entries"][0]
    old_path = ledger.project_root / entry["relative_path"]
    payload = json.loads(old_path.read_bytes())
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
    entry["content_hash"] = digest
    entry["relative_path"] = relative_path


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
    completion = _completion_evidence(ledger, attempt, outcome="accepted")
    record_completed_evidence_command(ledger.project_root, ledger, attempt, completion)

    if invalid_case == "evidence_cross_reference":
        _rewrite_payload(ledger, completion, lambda payload: payload["cross_references"].update({"policy_hash": "a" * 64}))
    elif invalid_case == "artifact_hash":
        (ledger.project_root / completion["entries"][0]["relative_path"]).write_text("{}", encoding="utf-8")
    elif invalid_case == "dataset_coverage":
        _rewrite_payload(ledger, completion, lambda payload: [row.__setitem__("dataset_id", "unregistered-dataset") for row in payload["rows"]])
    elif invalid_case == "phase_mismatch":
        _rewrite_payload(ledger, completion, lambda payload: [row.__setitem__("phase", "proxy") for row in payload["rows"]])
    else:
        completion["trial_spec_hash"] = "f" * 64

    _assert_precommit_rejected_without_authoritative_write(ledger, direction, completion)


def test_bootstrap_precommit_requires_verified_completion_before_finish_run(tmp_path: Path) -> None:
    ledger, direction, _, attempt = _prepared_attempt(tmp_path, bootstrap=True)
    completion = _completion_evidence(ledger, attempt, outcome="accepted")
    record_completed_evidence_command(ledger.project_root, ledger, attempt, completion)
    completion["entries"] = []
    _assert_precommit_rejected_without_authoritative_write(ledger, direction, completion)


def test_direction_aggregate_selects_best_accepted_delta_not_highest_absolute_candidate(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    expected_best_attempt_id = None
    rejected_high_candidate_attempt_id = None

    measurements = [(0.50, 0.70), (1.00, 0.95), (0.60, 0.68), (0.55, 0.57), (0.40, 0.46)]
    for index, (baseline, candidate) in enumerate(measurements, start=1):
        variant = _variant(direction, index)
        _initialize(ledger, direction, variant)
        attempt = _reserve(ledger, direction, variant)
        attempt = start_attempt_phase(ledger, attempt, "full")
        completion = _completion_evidence(ledger, attempt, outcome="accepted")
        def set_measurements(payload: dict) -> None:
            for row in payload["rows"]:
                row["metric_value"] = baseline if row["role"] == "baseline" else candidate
        _rewrite_payload(ledger, completion, set_measurements)
        record_completed_evidence_command(ledger.project_root, ledger, attempt, completion)
        ledger.complete_attempt(completion)
        if index == 1:
            expected_best_attempt_id = attempt["attempt_id"]
        if index == 2:
            rejected_high_candidate_attempt_id = attempt["attempt_id"]

    aggregate = ledger.state()["latest_direction_aggregate"]
    rejected = next(item for item in aggregate["outcomes"] if item["attempt_id"] == rejected_high_candidate_attempt_id)
    assert rejected["outcome"] == "rejected"
    assert aggregate["selection"]["best_attempt_id"] == expected_best_attempt_id
    assert aggregate["selection"]["best_attempt_id"] != rejected_high_candidate_attempt_id

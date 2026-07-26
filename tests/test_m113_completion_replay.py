from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sqlite3

import pytest

from auto_research.domain_contracts import canonical_hash
from auto_research.research_state import IntegrityError, ResearchEventLedger
from support.authoritative_evidence import record_completed_evidence_command
from test_authoritative_state_machine import _completion_evidence, _direction, _initialize, _reserve, _variant


def _ready_finalization(tmp_path: Path, *, outcome: str = "rejected"):
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    completion = _completion_evidence(ledger, attempt, outcome=outcome)
    authoritative_attempt = ledger.state()["attempts"][attempt["attempt_id"]]
    record_completed_evidence_command(tmp_path, ledger, authoritative_attempt, completion)
    completed, route = ledger.complete_attempt(completion)
    event = ledger.events()[-1]
    assert event["event_type"] == "AttemptFinalized"
    return ledger, direction, attempt, completion, completed, route, event


def _authoritative_snapshot(ledger: ResearchEventLedger, direction: dict, attempt_id: str) -> dict:
    state = ledger.state()
    return {
        "event_count": len(ledger.events()),
        "last_sequence": state["last_sequence"],
        "budget": deepcopy(state["directions"][direction["direction_semantic_hash"]]["budget"]),
        "trial_result": deepcopy(state["trial_results"][attempt_id]),
        "route": deepcopy(state["last_route_outcome"]),
        "aggregate": deepcopy(state["latest_direction_aggregate"]),
        "history": deepcopy(state["method_tried_history"]),
    }


def test_exact_completion_replay_returns_historical_result_without_writes(tmp_path: Path) -> None:
    ledger, direction, attempt, completion, completed, route, event = _ready_finalization(tmp_path)
    before = _authoritative_snapshot(ledger, direction, attempt["attempt_id"])

    replayed_attempt, replayed_route = ledger.complete_attempt(completion)
    after = _authoritative_snapshot(ledger, direction, attempt["attempt_id"])
    operation = ledger.query_operation_result(event["event_id"])

    assert replayed_attempt == completed
    assert replayed_route == route
    assert after == before
    assert canonical_hash(after["trial_result"]) == canonical_hash(before["trial_result"])
    assert replayed_route["source"]["event_id"] == event["event_id"]
    assert operation["aggregate"] == before["aggregate"]
    assert event["payload"]["request_fingerprint"] == ledger.events()[-1]["payload"]["request_fingerprint"]


def test_conflicting_completion_for_completed_attempt_is_zero_write_integrity_error(tmp_path: Path) -> None:
    ledger, direction, attempt, _, _, _, _ = _ready_finalization(tmp_path, outcome="rejected")
    conflicting = _completion_evidence(ledger, attempt, outcome="accepted")
    before = _authoritative_snapshot(ledger, direction, attempt["attempt_id"])

    with pytest.raises(
        IntegrityError,
        match="completion fingerprint conflict|receipt output hash|content hash differs",
    ):
        ledger.complete_attempt(conflicting)

    with sqlite3.connect(ledger.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == before["event_count"]
    assert _authoritative_snapshot(ledger, direction, attempt["attempt_id"]) == before


@pytest.mark.parametrize("attack", ["missing", "corrupt"])
def test_completion_replay_revalidates_immutable_evidence_before_returning_history(tmp_path: Path, attack: str) -> None:
    ledger, direction, attempt, completion, _, _, _ = _ready_finalization(tmp_path)
    before = _authoritative_snapshot(ledger, direction, attempt["attempt_id"])
    artifact = ledger.project_root / completion["entries"][0]["relative_path"]
    if attack == "missing":
        artifact.unlink()
        match = "artifact rejected|unavailable"
        state_match = "immutable receipt-bound evidence audit failed: evidence path contains a symlink or is unavailable"
    else:
        artifact.write_bytes(artifact.read_bytes() + b"\n")
        match = "evidence content hash mismatch|receipt output bytes differ|attempt-scoped evidence differs"
        state_match = (
            "immutable receipt-bound evidence audit failed: "
            "attempt-scoped evidence differs from deterministic derive output"
        )

    with pytest.raises(IntegrityError, match=match):
        ledger.complete_attempt(completion)

    with sqlite3.connect(ledger.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == before["event_count"]
    with pytest.raises(IntegrityError, match=state_match):
        ledger.state()

from pathlib import Path

from auto_research.research_state import ResearchEventLedger
from test_authoritative_state_machine import _completion_evidence, _direction, _variant, _initialize, _reserve, _complete


def test_immutable_events_rebuild_snapshot_and_attempt_view(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    _complete(ledger, attempt, outcome="rejected")

    events = ledger.events()
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert len({event["event_id"] for event in events}) == len(events)
    assert (tmp_path / "meta" / "attempts" / f"{attempt['attempt_id']}.json").exists()
    rebuilt = ResearchEventLedger(tmp_path).rebuild()
    assert rebuilt["directions"][direction["direction_semantic_hash"]]["budget"]["consumed"] == 1


def test_duplicate_attempt_completion_does_not_double_count(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    completion = _completion_evidence(ledger, attempt, outcome="rejected")
    ledger.transition_attempt(attempt["attempt_id"], "FULL_RUNNING", phase="full", phase_state="RUNNING")
    completed, _ = ledger.complete_attempt(completion)
    finalization_event = next(event for event in reversed(ledger.events()) if event["event_type"] == "AttemptFinalized")
    before = ledger.state()
    before_event_count = len(ledger.events())
    replayed, _ = ledger.complete_attempt(completion, event_id=finalization_event["event_id"])
    after = ledger.state()
    assert completed["attempt_id"] == attempt["attempt_id"]
    assert replayed == completed
    assert len(ledger.events()) == before_event_count
    assert after["last_sequence"] == before["last_sequence"]
    assert after["directions"][direction["direction_semantic_hash"]]["budget"] == {"target": 5, "reserved": 0, "consumed": 1}
    assert after["trial_results"] == before["trial_results"]
    assert after["last_route_outcome"] == before["last_route_outcome"]
    assert after["latest_direction_aggregate"] == before["latest_direction_aggregate"]

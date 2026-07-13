from pathlib import Path

from auto_research.research_state import ResearchEventLedger
from test_authoritative_state_machine import _direction, _variant, _initialize, _reserve, _complete


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
    assert rebuilt["directions"][direction["direction_hash"]]["budget"]["consumed"] == 1


def test_duplicate_attempt_completion_does_not_double_count(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    completed, _ = _complete(ledger, attempt, outcome="rejected")
    trial = ledger.state()["trial_results"][attempt["attempt_id"]]
    ledger.complete_attempt(trial)
    assert completed["attempt_id"] == attempt["attempt_id"]
    assert ledger.state()["directions"][direction["direction_hash"]]["budget"]["consumed"] == 1

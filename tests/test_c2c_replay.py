from pathlib import Path

from auto_research.c2c_e2e import build_c2c_replay_plan, build_c2c_replay_result
from auto_research.research_state import ResearchEventLedger
from test_authoritative_state_machine import _complete, _direction, _initialize, _reserve, _variant


def test_c2c_replay_rebuilds_generic_event_state(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    _complete(ledger, _reserve(ledger, direction, variant), outcome="rejected")

    plan = build_c2c_replay_plan(tmp_path)
    result = build_c2c_replay_result(tmp_path, plan)

    assert plan["expected_decision_source"] == "meta/research_events.sqlite3"
    assert result["status"] == "match"
    assert result["replayed_decisions"]["route_decision"] == "PROPOSE_NEXT_VARIANT"

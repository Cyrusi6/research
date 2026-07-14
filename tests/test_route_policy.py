from pathlib import Path

from auto_research.research_state import ResearchEventLedger
from test_authoritative_state_machine import _complete, _direction, _initialize, _reserve, _variant


def test_first_four_method_outcomes_route_to_next_variant(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    for index in range(1, 5):
        variant = _variant(direction, index)
        _initialize(ledger, direction, variant)
        _, route = _complete(ledger, _reserve(ledger, direction, variant), outcome="accepted" if index == 1 else "rejected")
        assert route["next_action"] == "PROPOSE_NEXT_VARIANT"


def test_fifth_without_acceptance_routes_to_new_direction(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    route = None
    for index in range(1, 6):
        variant = _variant(direction, index)
        _initialize(ledger, direction, variant)
        _, route = _complete(ledger, _reserve(ledger, direction, variant), outcome="rejected")
    assert route["next_action"] == "START_NEW_DIRECTION"


def test_resource_and_integrity_have_unified_actions(tmp_path: Path) -> None:
    for outcome, failure, expected in [
        ("resource_paused", "oom_retry", "PAUSE_RESOURCE"),
        ("integrity_blocked", "integrity", "BLOCK_INTEGRITY"),
    ]:
        root = tmp_path / expected
        ledger = ResearchEventLedger(root)
        direction = _direction()
        variant = _variant(direction, 1)
        _initialize(ledger, direction, variant)
        attempt = _reserve(ledger, direction, variant)
        ledger.transition_attempt(attempt["attempt_id"], "FULL_RUNNING", phase="full", phase_state="RUNNING")
        _, route = _complete(ledger, attempt, outcome=outcome, evaluable=False, failure=failure)
        assert route["next_action"] == expected

from pathlib import Path

from auto_research.research_state import ResearchEventLedger
from test_authoritative_state_machine import _complete, _direction, _failure_evidence, _initialize, _reserve, _variant
from support.authoritative_evidence import start_attempt_phase


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
    for failure_class, expected in [
        ("oom_retry", "PAUSE_RESOURCE"),
        ("integrity_failure", "BLOCK_INTEGRITY"),
    ]:
        root = tmp_path / expected
        ledger = ResearchEventLedger(root)
        direction = _direction()
        variant = _variant(direction, 1)
        _initialize(ledger, direction, variant)
        attempt = _reserve(ledger, direction, variant)
        attempt = start_attempt_phase(ledger, attempt, "full")
        _, route = ledger.disposition_failure(_failure_evidence(ledger, attempt, failure_class))
        assert route["next_action"] == expected

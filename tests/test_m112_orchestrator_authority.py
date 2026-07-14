from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import auto_research.config as config_module
import auto_research.orchestrator as orchestrator_module
from auto_research.agents import literature as literature_module
from auto_research.adapters.literature import LiteratureProvider
from auto_research.orchestrator import Orchestrator
from auto_research.research_state import IntegrityError, ResearchEventLedger
from test_authoritative_state_machine import _direction, _initialize, _reserve, _variant
from test_m113_ledger_closure import _failure_evidence as _canonical_failure_evidence
from test_m113_ledger_closure import _resume_evidence as _canonical_resume_evidence
from test_pipeline import _mock_generic_s1_codex, _test_config
from support.authoritative_evidence import start_attempt_phase


def _failure_evidence(ledger: ResearchEventLedger, attempt: dict, failure_class: str) -> dict:
    return _canonical_failure_evidence(
        ledger.project_root,
        attempt,
        failure_class=failure_class,
        exit_code=137 if failure_class == "resource_pause" else 1,
        resource_type="system_memory",
    )


def _routed_result(tmp_path: Path, *, variant_index: int = 1) -> tuple[ResearchEventLedger, dict, dict, int]:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, variant_index)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    attempt = start_attempt_phase(ledger, attempt, "full")
    routed_attempt, route = ledger.disposition_failure(_failure_evidence(ledger, attempt, "resource_pause"))
    source_event = next(event for event in ledger.events() if event["event_id"] == route["source"]["event_id"])
    return ledger, routed_attempt, route, source_event["sequence"]


def test_orchestrator_reads_route_from_committed_ledger_event(tmp_path: Path) -> None:
    ledger, attempt, route, sequence = _routed_result(tmp_path)

    authoritative = Orchestrator._authoritative_s3_route(
        tmp_path,
        {
            "attempt": attempt,
            "committed_event_id": route["source"]["event_id"],
            "committed_event_sequence": sequence,
            "route_outcome": deepcopy(route),
        },
    )

    assert authoritative == ledger.state()["last_route_outcome"]
    assert authoritative["source"]["attempt_id"] == attempt["attempt_id"]
    source_event = next(event for event in ledger.events() if event["event_id"] == authoritative["source"]["event_id"])
    assert source_event["event_type"] == "AttemptDispositioned"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda route: route["source"].update(event_id="missing-event"), "conflicts"),
        (lambda route: route["source"].update(attempt_id="other-attempt"), "conflicts"),
        (lambda route: route.update(next_action="FINISH_RUN"), "conflicts"),
        (lambda route: route["budget_snapshot"].update(consumed=5), "conflicts"),
        (lambda route: route["source"].update(sequence=999), "conflicts"),
        (lambda route: route.update(next_action="UNKNOWN_ACTION"), "conflicts"),
    ],
)
def test_orchestrator_rejects_diagnostic_route_conflicts(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    _, attempt, route, sequence = _routed_result(tmp_path)
    diagnostic = deepcopy(route)
    mutation(diagnostic)

    with pytest.raises(IntegrityError, match=message):
        Orchestrator._authoritative_s3_route(
            tmp_path,
            {
                "attempt": attempt,
                "committed_event_id": route["source"]["event_id"],
                "committed_event_sequence": sequence,
                "route_outcome": diagnostic,
            },
        )


def test_orchestrator_allows_missing_route_diagnostic(tmp_path: Path) -> None:
    _, attempt, route, sequence = _routed_result(tmp_path)

    assert Orchestrator._authoritative_s3_route(
        tmp_path,
        {
            "attempt": attempt,
            "committed_event_id": route["source"]["event_id"],
            "committed_event_sequence": sequence,
        },
    ) == route


def test_orchestrator_rejects_wrong_committed_sequence(tmp_path: Path) -> None:
    _, attempt, route, sequence = _routed_result(tmp_path)

    with pytest.raises(IntegrityError, match="sequence mismatch"):
        Orchestrator._authoritative_s3_route(
            tmp_path,
            {
                "attempt": attempt,
                "committed_event_id": route["source"]["event_id"],
                "committed_event_sequence": sequence + 1,
            },
        )


def test_orchestrator_rejects_late_result_after_newer_attempt_route(tmp_path: Path) -> None:
    ledger, first_attempt, first_route, first_sequence = _routed_result(tmp_path)
    first_attempt = ledger.resume_attempt(
        _canonical_resume_evidence(tmp_path, ledger, first_attempt, resource_type="system_memory")
    )
    current_attempt = start_attempt_phase(ledger, first_attempt, "full")
    ledger.disposition_failure(_failure_evidence(ledger, current_attempt, "activation_failure"))

    with pytest.raises(IntegrityError, match="stale S3 result"):
        Orchestrator._authoritative_s3_route(
            tmp_path,
            {
                "attempt": first_attempt,
                "committed_event_id": first_route["source"]["event_id"],
                "committed_event_sequence": first_sequence,
                "route_outcome": first_route,
            },
        )


def test_real_simulated_pipeline_uses_ledger_route_not_result_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _test_config(tmp_path, simulate=True)
    monkeypatch.setattr(config_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)
    monkeypatch.setattr(LiteratureProvider, "search", lambda self, topic: [])
    monkeypatch.setattr(LiteratureProvider, "download_pdf", lambda self, url: None)
    _mock_generic_s1_codex(monkeypatch)

    orchestrator = Orchestrator()
    project_id = orchestrator.init_project("M1.1.2 route authority", project_id="m112-route-e2e", simulate=True)
    result = orchestrator.start(project_id)

    assert result["status"] == "completed", result
    project_root = tmp_path / project_id
    ledger = ResearchEventLedger(project_root)
    state = ledger.state()
    route = state["last_route_outcome"]
    source_event = next(event for event in ledger.events() if event["event_id"] == route["source"]["event_id"])
    assert route["source"]["attempt_id"] == source_event["payload"]["trial_result"]["attempt_id"]
    assert route["next_action"] == "FINISH_DIRECTION"
    assert state["directions"][route["identity"]["direction_semantic_hash"]]["budget"] == {
        "target": 5,
        "reserved": 0,
        "consumed": 5,
    }

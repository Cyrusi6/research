from __future__ import annotations

from pathlib import Path

import pytest

from auto_research.agents.experiment import ExperimentAgent
from auto_research.contract_store import ContractStore
from auto_research.research_state import ResearchEventLedger
from test_m1151_command_restart import _install_counted_producer
from test_m115_final_e2e import _generic_project


def _events(root: Path) -> list[dict]:
    ledger = ResearchEventLedger(root)
    events = ledger.events()
    assert ledger.rebuild()["last_sequence"] == len(events)
    return events


def _event_count(root: Path, event_type: str) -> int:
    return sum(event["event_type"] == event_type for event in _events(root))


def _command_refs(root: Path) -> dict[str, tuple[str, tuple[str, ...]]]:
    state = ResearchEventLedger(root).state()
    store = ContractStore(root)
    references = {}
    for command_id, record in state["phase_commands"].items():
        if record["status"] != "completed":
            continue
        receipt = store.read_json(record["receipt_ref"], schema_file="phase_run_receipt_v5.schema.json")
        references[command_id] = (
            record["receipt_ref"]["digest"],
            tuple(output["content_hash"] for output in receipt["outputs"]),
        )
    return references


def _direction_budget(root: Path) -> dict[str, int]:
    state = ResearchEventLedger(root).state()
    direction = next(iter(state["directions"].values()))
    return direction["budget"]




def test_restart_after_started_before_receipt_blocks_without_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "started-before-receipt"
    root.mkdir()
    context, _, _ = _generic_project(root)
    marker = _install_counted_producer(root)
    first = ExperimentAgent(context)
    run_step = first.runner.run_step

    def crash_after_side_effect(**kwargs):
        run_step(**kwargs)
        raise RuntimeError("crash after side effect before durable receipt")

    monkeypatch.setattr(first.runner, "run_step", crash_after_side_effect)
    with pytest.raises(RuntimeError, match="before durable receipt"):
        first.run()

    assert marker.read_text(encoding="utf-8").splitlines() == ["invoked"]
    assert _event_count(root, "PhaseCommandStarted") == 1
    assert _event_count(root, "PhaseCommandUnknownOutcome") == 1
    before_sequence = ResearchEventLedger(root).state()["last_sequence"]

    restarted = ExperimentAgent(context)
    monkeypatch.setattr(
        restarted.runner,
        "run_step",
        lambda **kwargs: pytest.fail(f"unknown command was rerun: {kwargs}"),
    )
    result = restarted.run()

    assert result["route_outcome"]["next_action"] == "BLOCK_INTEGRITY"
    assert marker.read_text(encoding="utf-8").splitlines() == ["invoked"]
    assert ResearchEventLedger(root).state()["last_sequence"] == before_sequence
    assert _direction_budget(root) == {"target": 5, "reserved": 0, "consumed": 0}


def test_restart_after_durable_receipt_before_completed_reconciles_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "receipt-before-completed"
    root.mkdir()
    context, _, _ = _generic_project(root)
    marker = _install_counted_producer(root)
    original = ResearchEventLedger.complete_phase_command
    crashed = False

    def crash_before_completed(self, command_id, receipt_ref):
        nonlocal crashed
        if not crashed:
            crashed = True
            raise RuntimeError("crash after durable receipt before PhaseCommandCompleted")
        return original(self, command_id, receipt_ref)

    monkeypatch.setattr(ResearchEventLedger, "complete_phase_command", crash_before_completed)
    with pytest.raises(RuntimeError, match="before PhaseCommandCompleted"):
        ExperimentAgent(context).run()

    assert marker.read_text(encoding="utf-8").splitlines() == ["invoked"]
    assert _event_count(root, "PhaseCommandStarted") == 1
    assert _event_count(root, "PhaseCommandCompleted") == 0
    assert list((root / "meta" / "command_receipts").glob("*.json"))

    result = ExperimentAgent(context).run()

    assert result["route_outcome"]["next_action"] == "PROPOSE_NEXT_VARIANT"
    assert marker.read_text(encoding="utf-8").splitlines() == ["invoked"]
    assert _event_count(root, "PhaseCommandStarted") == 2
    assert _event_count(root, "PhaseCommandCompleted") == 2
    assert _event_count(root, "AttemptFinalized") == 1
    assert _direction_budget(root) == {"target": 5, "reserved": 0, "consumed": 1}


def test_restart_after_completed_before_evidence_commit_reuses_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "completed-before-evidence"
    root.mkdir()
    context, _, _ = _generic_project(root)
    marker = _install_counted_producer(root)
    first = ExperimentAgent(context)
    execute_phase = first._execute_phase

    def crash_after_completed(*args, **kwargs):
        execute_phase(*args, **kwargs)
        raise RuntimeError("crash after PhaseCommandCompleted before evidence commit")

    monkeypatch.setattr(first, "_execute_phase", crash_after_completed)
    with pytest.raises(RuntimeError, match="before evidence commit"):
        first.run()

    receipt_refs = _command_refs(root)
    assert receipt_refs
    assert marker.read_text(encoding="utf-8").splitlines() == ["invoked"]
    assert _event_count(root, "AttemptFinalized") == 0

    result = ExperimentAgent(context).run()

    assert result["route_outcome"]["next_action"] == "PROPOSE_NEXT_VARIANT"
    assert marker.read_text(encoding="utf-8").splitlines() == ["invoked"]
    assert _command_refs(root) == receipt_refs
    assert _event_count(root, "AttemptFinalized") == 1
    assert _direction_budget(root) == {"target": 5, "reserved": 0, "consumed": 1}


def test_restart_after_full_completed_before_finalization_preserves_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "full-completed-before-finalization"
    root.mkdir()
    context, _, _ = _generic_project(root)
    marker = _install_counted_producer(root)
    first = ExperimentAgent(context)

    def crash_before_finalization(*args, **kwargs):
        raise RuntimeError("crash after full receipt before AttemptFinalized")

    monkeypatch.setattr(first, "_finalize_trial", crash_before_finalization)
    with pytest.raises(RuntimeError, match="before AttemptFinalized"):
        first.run()
    receipt_refs = _command_refs(root)
    before_sequence = ResearchEventLedger(root).state()["last_sequence"]

    result = ExperimentAgent(context).run()

    assert result["route_outcome"]["next_action"] == "PROPOSE_NEXT_VARIANT"
    assert marker.read_text(encoding="utf-8").splitlines() == ["invoked"]
    assert _command_refs(root) == receipt_refs
    assert ResearchEventLedger(root).state()["last_sequence"] > before_sequence
    assert _event_count(root, "AttemptFinalized") == 1
    assert _direction_budget(root) == {"target": 5, "reserved": 0, "consumed": 1}


def test_restart_after_finalization_before_route_delivery_returns_historical_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "finalized-before-route"
    root.mkdir()
    context, _, _ = _generic_project(root)
    marker = _install_counted_producer(root)
    original = ResearchEventLedger.complete_attempt
    crashed = False

    def crash_after_finalization(self, completion_evidence, *, event_id=None):
        nonlocal crashed
        committed = original(self, completion_evidence, event_id=event_id)
        if not crashed:
            crashed = True
            raise RuntimeError("crash after AttemptFinalized before route delivery")
        return committed

    monkeypatch.setattr(ResearchEventLedger, "complete_attempt", crash_after_finalization)
    with pytest.raises(RuntimeError, match="before route delivery"):
        ExperimentAgent(context).run()

    ledger = ResearchEventLedger(root)
    before = ledger.state()
    final_event = next(event for event in reversed(ledger.events()) if event["event_type"] == "AttemptFinalized")
    historical = ledger.query_operation_result(final_event["event_id"])
    assert historical["route_outcome"] is not None
    assert marker.read_text(encoding="utf-8").splitlines() == ["invoked"]

    replay = ExperimentAgent(context).run()
    after = ResearchEventLedger(root).state()

    assert replay["route_outcome"] == historical["route_outcome"]
    assert replay["trial_result"] == historical["trial_result"]
    assert marker.read_text(encoding="utf-8").splitlines() == ["invoked"]
    assert after["last_sequence"] == before["last_sequence"]
    assert after["trial_results"] == before["trial_results"]
    assert _event_count(root, "AttemptFinalized") == 1
    assert _direction_budget(root) == {"target": 5, "reserved": 0, "consumed": 1}

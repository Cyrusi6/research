from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from auto_research.agents.experiment import ExperimentAgent
from auto_research.research_state import IntegrityError, ResearchEventLedger
from auto_research.validators import run_stage_gate
from support.m1152_local_subprocess import (
    activate_c2c_variant,
    activate_generic_variant,
    assert_trial_lineage,
    c2c_variant,
    c2c_invocations,
    command_lineage,
    create_c2c_project,
    create_generic_project,
    direction_budget,
    event_types,
    generic_invocations,
    generic_variant,
)


def _event_count(root: Path, event_type: str) -> int:
    return event_types(root).count(event_type)


def _record_counter(records: list[dict]) -> Counter[str]:
    return Counter(json.dumps(record, sort_keys=True) for record in records)


def _full_c2c_records(records: list[dict]) -> list[dict]:
    return [
        record
        for record in records
        if record.get("kind") in {"train", "eval"}
        and "/proxy/" not in str(record.get("config") or "")
        and "proxy_baseline" not in str(record.get("config") or "")
    ]


def _assert_gate_passes(root: Path, config: dict) -> None:
    first = run_stage_gate("S3_experiment", root, config).to_dict()
    before = ResearchEventLedger(root).state()["last_sequence"]
    second = run_stage_gate("S3_experiment", root, config).to_dict()
    assert first["status"] == "PASS", first
    assert second["status"] == "PASS", second
    assert ResearchEventLedger(root).state()["last_sequence"] == before


def test_crash_after_started_before_receipt_blocks_without_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "started-before-receipt"
    root.mkdir()
    context, _, _ = create_generic_project(root)
    first = ExperimentAgent(context)
    actual_run_step = first.runner.run_step

    def crash_after_side_effect(**kwargs):
        actual_run_step(**kwargs)
        raise RuntimeError("crash after side effect before durable receipt")

    monkeypatch.setattr(first.runner, "run_step", crash_after_side_effect)
    with pytest.raises(RuntimeError, match="before durable receipt"):
        first.run()

    assert len(generic_invocations(root)) == 1
    assert _event_count(root, "PhaseCommandStarted") == 1
    assert _event_count(root, "PhaseCommandUnknownOutcome") == 1
    before = ResearchEventLedger(root).state()

    restarted = ExperimentAgent(context)
    monkeypatch.setattr(
        restarted.runner,
        "run_step",
        lambda **kwargs: pytest.fail(f"unknown command was rerun: {kwargs}"),
    )
    result = restarted.run()

    assert result["route_outcome"]["next_action"] == "BLOCK_INTEGRITY"
    assert len(generic_invocations(root)) == 1
    assert ResearchEventLedger(root).state()["last_sequence"] == before["last_sequence"]
    assert direction_budget(root) == {"target": 5, "reserved": 0, "consumed": 0}


def test_crash_after_durable_receipt_before_completed_reconciles_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "receipt-before-completed"
    root.mkdir()
    context, _, _ = create_generic_project(root)
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

    assert len(generic_invocations(root)) == 1
    assert _event_count(root, "PhaseCommandStarted") == 1
    assert _event_count(root, "PhaseCommandCompleted") == 0
    locators = list((root / "meta" / "command_receipts").glob("*.json"))
    assert len(locators) == 1

    result = ExperimentAgent(context).run()

    assert result["route_outcome"]["next_action"] == "PROPOSE_NEXT_VARIANT"
    assert len(generic_invocations(root)) == 1
    assert _event_count(root, "PhaseCommandStarted") == 1
    assert _event_count(root, "PhaseCommandCompleted") == 1
    assert _event_count(root, "AttemptFinalized") == 1
    assert direction_budget(root) == {"target": 5, "reserved": 0, "consumed": 1}
    assert_trial_lineage(root)


def test_crash_after_completed_before_evidence_commit_reuses_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "completed-before-evidence"
    root.mkdir()
    context, _, _ = create_generic_project(root)
    first = ExperimentAgent(context)
    execute_phase = first._execute_phase

    def crash_after_completed(*args, **kwargs):
        execute_phase(*args, **kwargs)
        raise RuntimeError("crash after PhaseCommandCompleted before evidence commit")

    monkeypatch.setattr(first, "_execute_phase", crash_after_completed)
    with pytest.raises(RuntimeError, match="before evidence commit"):
        first.run()

    before_lineage = command_lineage(root)
    assert before_lineage
    assert len(generic_invocations(root)) == 1
    assert _event_count(root, "AttemptFinalized") == 0

    result = ExperimentAgent(context).run()

    assert result["route_outcome"]["next_action"] == "PROPOSE_NEXT_VARIANT"
    assert len(generic_invocations(root)) == 1
    assert command_lineage(root) == before_lineage
    assert _event_count(root, "AttemptFinalized") == 1
    assert direction_budget(root) == {"target": 5, "reserved": 0, "consumed": 1}
    assert_trial_lineage(root)


def test_crash_after_proxy_commit_before_full_start_reuses_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, context, _ = create_c2c_project(
        tmp_path,
        monkeypatch,
        profile="standard",
        proxy_accuracy=0.51,
        name="proxy-commit-before-full",
    )
    original = ResearchEventLedger.start_full_phase
    crashed = False

    def crash_before_full(self, attempt_id, **kwargs):
        nonlocal crashed
        if not crashed:
            crashed = True
            raise RuntimeError("crash after ProxyEvidenceCommitted before FullPhaseStarted")
        return original(self, attempt_id, **kwargs)

    monkeypatch.setattr(ResearchEventLedger, "start_full_phase", crash_before_full)
    with pytest.raises(RuntimeError, match="before FullPhaseStarted"):
        ExperimentAgent(context).run()

    proxy_records = _record_counter(c2c_invocations(root))
    proxy_lineage = command_lineage(root)
    assert _event_count(root, "ProxyEvidenceCommitted") == 1
    assert _event_count(root, "FullPhaseStarted") == 0
    assert not _full_c2c_records(c2c_invocations(root))
    assert direction_budget(root) == {"target": 5, "reserved": 1, "consumed": 0}

    result = ExperimentAgent(context).run()

    assert result["route_outcome"]["next_action"] == "PROPOSE_NEXT_VARIANT"
    after_records = _record_counter(c2c_invocations(root))
    for record, count in proxy_records.items():
        assert after_records[record] == count
    for command_id, hashes in proxy_lineage.items():
        assert command_lineage(root)[command_id] == hashes
    assert len(_full_c2c_records(c2c_invocations(root))) == 3
    assert _event_count(root, "ProxyEvidenceCommitted") == 1
    assert _event_count(root, "FullPhaseStarted") == 1
    assert _event_count(root, "AttemptFinalized") == 1
    assert direction_budget(root) == {"target": 5, "reserved": 0, "consumed": 1}
    assert_trial_lineage(root)


def test_crash_after_full_started_before_first_command_runs_full_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, context, _ = create_c2c_project(
        tmp_path,
        monkeypatch,
        profile="standard",
        proxy_accuracy=0.51,
        name="full-started-before-command",
    )
    first = ExperimentAgent(context)
    execute_phase = first._execute_phase

    def crash_before_full_command(*args, **kwargs):
        if first._active_phase_context.phase == "full":
            raise RuntimeError("crash after FullPhaseStarted before first full command")
        return execute_phase(*args, **kwargs)

    monkeypatch.setattr(first, "_execute_phase", crash_before_full_command)
    with pytest.raises(RuntimeError, match="before first full command"):
        first.run()

    proxy_records = _record_counter(c2c_invocations(root))
    assert _event_count(root, "FullPhaseStarted") == 1
    assert not _full_c2c_records(c2c_invocations(root))

    result = ExperimentAgent(context).run()

    assert result["route_outcome"]["next_action"] == "PROPOSE_NEXT_VARIANT"
    after_records = _record_counter(c2c_invocations(root))
    for record, count in proxy_records.items():
        assert after_records[record] == count
    assert len(_full_c2c_records(c2c_invocations(root))) == 3
    assert _event_count(root, "FullPhaseStarted") == 1
    assert _event_count(root, "AttemptFinalized") == 1
    assert direction_budget(root) == {"target": 5, "reserved": 0, "consumed": 1}
    assert_trial_lineage(root)


def test_crash_after_full_commands_before_finalization_reuses_all_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, context, _ = create_c2c_project(
        tmp_path,
        monkeypatch,
        profile="standard",
        proxy_accuracy=0.51,
        name="full-completed-before-finalization",
    )
    first = ExperimentAgent(context)
    finalize = first._finalize_trial

    def crash_before_finalization(result, *, attempt, trial_spec, ledger):
        if attempt["state"] == "FULL_RUNNING":
            raise RuntimeError("crash after full command completion before AttemptFinalized")
        return finalize(result, attempt=attempt, trial_spec=trial_spec, ledger=ledger)

    monkeypatch.setattr(first, "_finalize_trial", crash_before_finalization)
    with pytest.raises(RuntimeError, match="before AttemptFinalized"):
        first.run()

    before_records = _record_counter(c2c_invocations(root))
    before_lineage = command_lineage(root)
    assert len(_full_c2c_records(c2c_invocations(root))) == 3
    assert _event_count(root, "AttemptFinalized") == 0

    result = ExperimentAgent(context).run()

    assert result["route_outcome"]["next_action"] == "PROPOSE_NEXT_VARIANT"
    assert _record_counter(c2c_invocations(root)) == before_records
    assert command_lineage(root) == before_lineage
    assert _event_count(root, "AttemptFinalized") == 1
    assert direction_budget(root) == {"target": 5, "reserved": 0, "consumed": 1}
    assert_trial_lineage(root)


def test_crash_after_finalization_before_route_delivery_returns_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "finalized-before-route"
    root.mkdir()
    context, _, _ = create_generic_project(root)
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
    finalized = next(event for event in reversed(ledger.events()) if event["event_type"] == "AttemptFinalized")
    historical = ledger.query_operation_result(finalized["event_id"])
    before_lineage = command_lineage(root)
    assert len(generic_invocations(root)) == 1

    replay = ExperimentAgent(context).run()
    after = ResearchEventLedger(root).state()

    assert replay["route_outcome"] == historical["route_outcome"]
    assert replay["trial_result"] == historical["trial_result"]
    assert len(generic_invocations(root)) == 1
    assert command_lineage(root) == before_lineage
    assert after["last_sequence"] == before["last_sequence"]
    assert after["trial_results"] == before["trial_results"]
    assert _event_count(root, "AttemptFinalized") == 1
    assert direction_budget(root) == {"target": 5, "reserved": 0, "consumed": 1}


def test_generic_non_simulated_single_attempt_uses_receipt_derivation_and_gate(tmp_path: Path) -> None:
    root = tmp_path / "generic-single"
    root.mkdir()
    context, _, _ = create_generic_project(root)

    result = ExperimentAgent(context).run()

    assert result["route_outcome"]["next_action"] == "PROPOSE_NEXT_VARIANT"
    assert len(generic_invocations(root)) == 1
    assert event_types(root).index("PhaseCommandCompleted") < event_types(root).index("AttemptFinalized")
    assert direction_budget(root) == {"target": 5, "reserved": 0, "consumed": 1}
    assert_trial_lineage(root)
    _assert_gate_passes(root, context.config)


@pytest.mark.parametrize(
    ("proxy_accuracy", "expects_full", "consumed"),
    [(0.49, False, 0), (0.51, True, 1)],
)
def test_c2c_non_simulated_single_attempt_obeys_proxy_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proxy_accuracy: float,
    expects_full: bool,
    consumed: int,
) -> None:
    root, _, context, _ = create_c2c_project(
        tmp_path,
        monkeypatch,
        profile="standard",
        proxy_accuracy=proxy_accuracy,
        name=f"c2c-single-{proxy_accuracy}",
    )

    result = ExperimentAgent(context).run()
    events = event_types(root)
    records = c2c_invocations(root)

    assert result["route_outcome"]["next_action"] == "PROPOSE_NEXT_VARIANT"
    assert "ProxyEvidenceCommitted" in events
    assert bool(_full_c2c_records(records)) is expects_full
    if expects_full:
        assert events.index("ProxyEvidenceCommitted") < events.index("FullPhaseStarted")
        assert events.index("FullPhaseStarted") < events.index("AttemptFinalized")
        assert_trial_lineage(root)
    else:
        assert "FullPhaseStarted" not in events
        assert "AttemptFinalized" not in events
    assert direction_budget(root) == {"target": 5, "reserved": 0, "consumed": consumed}
    _assert_gate_passes(root, context.config)


def test_c2c_non_simulated_bootstrap_is_proxy_only_and_budget_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, context, _ = create_c2c_project(
        tmp_path,
        monkeypatch,
        profile="bootstrap",
        proxy_accuracy=0.51,
        name="c2c-bootstrap",
    )

    result = ExperimentAgent(context).run()
    state = ResearchEventLedger(root).state()

    assert result["route_outcome"]["next_action"] == "FINISH_RUN"
    assert "FullPhaseStarted" not in event_types(root)
    assert not _full_c2c_records(c2c_invocations(root))
    assert direction_budget(root) == {"target": 5, "reserved": 0, "consumed": 0}
    assert state["method_tried_history"] == []
    assert state["latest_direction_aggregate"] is None
    assert len(state["trial_results"]) == 1
    assert_trial_lineage(root)
    _assert_gate_passes(root, context.config)


def test_generic_non_simulated_five_variants_and_sixth_precommand_rejection(tmp_path: Path) -> None:
    root = tmp_path / "generic-five"
    root.mkdir()
    context, direction, _ = create_generic_project(root)
    routes: list[str] = []
    semantic_hashes: list[str] = []

    for index in range(1, 6):
        variant = activate_generic_variant(root, direction, index)
        semantic_hashes.append(variant["variant_semantic_hash"])
        routes.append(ExperimentAgent(context).run()["route_outcome"]["next_action"])

    ledger = ResearchEventLedger(root)
    state = ledger.state()
    assert len(set(semantic_hashes)) == 5
    assert routes[:4] == ["PROPOSE_NEXT_VARIANT"] * 4
    assert routes[4] == "FINISH_DIRECTION"
    assert direction_budget(root) == {"target": 5, "reserved": 0, "consumed": 5}
    assert len(state["trial_results"]) == 5
    assert len(generic_invocations(root)) == 5
    assert_trial_lineage(root)

    before_sequence = state["last_sequence"]
    before_commands = len(state["phase_commands"])
    before_contracts = {path.relative_to(root) for path in (root / "meta" / "contracts").rglob("*.json")}
    before_invocations = list(generic_invocations(root))
    sixth = generic_variant(direction, 6)
    assert sixth["variant_semantic_hash"] not in semantic_hashes
    with pytest.raises(IntegrityError, match="closed direction"):
        ledger.select_direction(direction)
    after = ResearchEventLedger(root).state()
    assert after["last_sequence"] == before_sequence
    assert len(after["phase_commands"]) == before_commands
    assert generic_invocations(root) == before_invocations
    assert {path.relative_to(root) for path in (root / "meta" / "contracts").rglob("*.json")} == before_contracts


def test_c2c_non_simulated_five_variants_and_sixth_precommand_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, context, direction = create_c2c_project(
        tmp_path,
        monkeypatch,
        profile="standard",
        proxy_accuracy=0.51,
        name="c2c-five",
    )
    routes: list[str] = []
    semantic_hashes: list[str] = []

    for index in range(1, 6):
        variant = activate_c2c_variant(root, direction, index)
        semantic_hashes.append(variant["variant_semantic_hash"])
        routes.append(ExperimentAgent(context).run()["route_outcome"]["next_action"])

    ledger = ResearchEventLedger(root)
    state = ledger.state()
    assert len(set(semantic_hashes)) == 5
    assert routes[:4] == ["PROPOSE_NEXT_VARIANT"] * 4
    assert routes[4] == "FINISH_DIRECTION"
    assert direction_budget(root) == {"target": 5, "reserved": 0, "consumed": 5}
    assert len(state["trial_results"]) == 5
    assert len(_full_c2c_records(c2c_invocations(root))) == 15
    for attempt in state["attempts"].values():
        assert attempt["phase_executions"]["proxy"]["phase_start_event_id"]
        assert attempt["phase_executions"]["full"]["phase_start_event_id"]
    assert_trial_lineage(root)

    before_sequence = state["last_sequence"]
    before_commands = len(state["phase_commands"])
    before_contracts = {path.relative_to(root) for path in (root / "meta" / "contracts").rglob("*.json")}
    before_invocations = _record_counter(c2c_invocations(root))
    sixth = c2c_variant(direction, 6)
    assert sixth["variant_semantic_hash"] not in semantic_hashes
    with pytest.raises(IntegrityError, match="closed direction"):
        ledger.select_direction(direction)
    after = ResearchEventLedger(root).state()
    assert after["last_sequence"] == before_sequence
    assert len(after["phase_commands"]) == before_commands
    assert _record_counter(c2c_invocations(root)) == before_invocations
    assert {path.relative_to(root) for path in (root / "meta" / "contracts").rglob("*.json")} == before_contracts

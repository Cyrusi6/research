from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

from auto_research.agents.experiment import ExperimentAgent
from auto_research.command_journal import CommandJournalError, CommandJournalResult, LedgerCommandJournal
from auto_research.phase_execution import GenericExternalPhaseExecutor, TypedPhaseFailure, require_executor_capability
from auto_research.phase_command_plan import build_phase_command_plan, store_phase_command_plan
from auto_research.research_state import IntegrityError, ResearchEventLedger
from test_m115_contract_journal import FakeLedger, _command_context, _command_result
from test_m115_final_e2e import _GENERIC_PRODUCER, _generic_project
from test_m115_phase_executor import Authority, _authorization, _context, _inventory


def _install_counted_producer(root: Path) -> Path:
    marker = root / "runner" / "producer-invocations.log"
    marker_setup = (
        "root=Path('.')\n"
        "marker=root/'runner/producer-invocations.log'\n"
        "marker.parent.mkdir(parents=True,exist_ok=True)\n"
        "with marker.open('a',encoding='utf-8') as stream: stream.write('invoked\\n')"
    )
    script = _GENERIC_PRODUCER.replace("root=Path('.')", marker_setup, 1)
    (root / "producer.py").write_text(script, encoding="utf-8")
    return marker


def _command_event_count(root: Path) -> int:
    return sum(
        event["event_type"] in {"PhaseCommandStarted", "PhaseCommandCompleted"}
        for event in ResearchEventLedger(root).events()
    )


def _with_frozen_command_plan(root: Path, context, authorization):
    plan = build_phase_command_plan(
        phase="full",
        adapter_id=authorization.adapter_identity,
        adapter_version="1",
        provenance_mode="local-external",
        variant_spec_hash=context.variant_spec_hash,
        source_snapshot_hash="f" * 64,
        command_values=({
            "argv": ["python", "producer.py"],
            "cwd": "work",
            "physical_raw_outputs": [{
                "output_id": "raw-main-results",
                "kind": "raw_main_results",
                "schema_version": "auto_research_main_results_v3",
                "locator": "raw/main-results.json",
                "locator_type": "file",
                "dataset_id": None,
                "role": None,
                "required": True,
                "normalized_kinds": ["main_results"],
            }],
        },),
        expected_evidence=({
            "kind": "main_results",
            "schema_version": "auto_research_main_results_v3",
            "required": True,
        },),
        default_cwd="work",
        project_root=root,
        coverage_contract={
            "mode": "exact_cartesian",
            "datasets": ["dataset-a"],
            "seeds": [0],
            "metrics": ["accuracy"],
            "roles": ["candidate"],
        },
    )
    _, plan_hash = store_phase_command_plan(root, plan)
    authorization = replace(authorization, command_plan_hash=plan_hash)
    context = replace(context, command_plan_hash=plan_hash, authorization_hash=authorization.authorization_hash)
    return context, authorization


def test_generic_restart_after_command_completed_recovers_receipt_without_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "generic-command-restart"
    root.mkdir()
    context, _, _ = _generic_project(root)
    marker = _install_counted_producer(root)

    first_agent = ExperimentAgent(context)

    def crash_before_finalization(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("crash after PhaseCommandCompleted before AttemptFinalized")

    monkeypatch.setattr(first_agent, "_finalize_trial", crash_before_finalization)
    with pytest.raises(RuntimeError, match="crash after PhaseCommandCompleted"):
        first_agent.run()

    ledger = ResearchEventLedger(root)
    event_types = [event["event_type"] for event in ledger.events()]
    assert event_types.count("PhaseCommandStarted") == 2
    assert event_types.count("PhaseCommandCompleted") == 2
    assert "AttemptFinalized" not in event_types
    assert marker.read_text(encoding="utf-8").splitlines() == ["invoked"]

    (root / "plan/trial_spec.json").write_text("{}\n", encoding="utf-8")

    restarted_agent = ExperimentAgent(context)
    monkeypatch.setattr(
        restarted_agent,
        "_run_generic_external_phase",
        lambda *args, **kwargs: pytest.fail("completed receipt replay re-entered the mutable adapter callback"),
    )
    try:
        restarted_result = restarted_agent.run()
    except IntegrityError as error:
        assert marker.read_text(encoding="utf-8").splitlines() == ["invoked"]
        assert _command_event_count(root) == 4
        pytest.fail(f"restart could not recover the committed receipt: {error}")

    assert marker.read_text(encoding="utf-8").splitlines() == ["invoked"]
    assert _command_event_count(root) == 4
    assert restarted_result["route_outcome"]["next_action"] == "PROPOSE_NEXT_VARIANT"
    assert [event["event_type"] for event in ledger.events()].count("AttemptFinalized") == 1


def test_synthetic_execution_uses_ledger_command_journal(tmp_path: Path) -> None:
    root = tmp_path / "synthetic-command-journal"
    root.mkdir()
    context, _, _ = _generic_project(root)
    context.config["experiment"]["simulate"] = True

    ExperimentAgent(context).run()

    events = ResearchEventLedger(root).events()
    assert [event["event_type"] for event in events].count("PhaseCommandStarted") == 2
    assert [event["event_type"] for event in events].count("PhaseCommandCompleted") == 2


def test_authoritative_step_rejects_command_not_in_frozen_plan_before_event_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "generic-command-injection"
    root.mkdir()
    context, _, _ = _generic_project(root)
    agent = ExperimentAgent(context)

    class ExpectedStop(RuntimeError):
        pass

    def inject_unplanned_command(execution, env_record_path, attempt):
        del execution, env_record_path, attempt
        ledger = ResearchEventLedger(root)
        before_sequence = ledger.state()["last_sequence"]
        before_commands = _command_event_count(root)
        try:
            agent._run_authoritative_step(
                name="command_0",
                command=[sys.executable, "-c", "pass"],
                working_dir=root,
            )
        except (IntegrityError, CommandJournalError):
            after = ledger.state()
            assert after["last_sequence"] == before_sequence
            assert _command_event_count(root) == before_commands
            raise ExpectedStop("unplanned command rejected before PhaseCommandStarted")
        raise AssertionError(
            "unplanned python -c pass was accepted despite frozen producer.py; "
            f"command events changed from {before_commands} to {_command_event_count(root)}"
        )

    monkeypatch.setattr(agent, "_run_generic_external_phase", inject_unplanned_command)
    with pytest.raises(ExpectedStop, match="rejected before PhaseCommandStarted"):
        agent.run()


def test_journal_reconciles_durable_receipt_and_recovers_typed_inventory(tmp_path: Path) -> None:
    context, authorization = _command_context(tmp_path)
    context, authorization = _with_frozen_command_plan(tmp_path, context, authorization)
    ledger = FakeLedger(authorization)
    ledger.complete_error = True
    journal = LedgerCommandJournal(tmp_path, ledger)
    calls: list[str] = []
    kwargs = dict(
        command_id="command-full-restart",
        argv=("python", "producer.py"),
        cwd="work",
        source_snapshot_hash="f" * 64,
        expected_outputs=(),
        runner=lambda: calls.append("run") or _command_result(tmp_path),
    )

    with pytest.raises(RuntimeError, match="DB crash"):
        journal.run_once(context, **kwargs)
    assert calls == ["run"]

    ledger.complete_error = False
    recovered = journal.run_once(context, **kwargs)
    assert isinstance(recovered, CommandJournalResult)
    assert recovered.execution_result.exit_code == 0
    assert recovered.execution_result.outputs == ()
    assert recovered.execution_result.raw_outputs[0]["kind"] == "raw_main_results"
    assert recovered.execution_result.raw_outputs[0]["content_hash"] == _command_result(tmp_path).raw_outputs[0]["contract_ref"]["digest"]
    assert recovered.artifact_inventory.context == context
    assert recovered.artifact_inventory.artifacts == ()
    assert recovered.artifact_inventory.complete is False
    assert calls == ["run"]

    replayed = journal.run_once(context, **kwargs)
    assert isinstance(replayed, CommandJournalResult)
    assert replayed.receipt_ref == recovered.receipt_ref
    assert replayed.execution_result == recovered.execution_result
    assert replayed.artifact_inventory == recovered.artifact_inventory
    assert calls == ["run"]


def test_started_without_durable_receipt_is_never_rerun(tmp_path: Path) -> None:
    context, authorization = _command_context(tmp_path)
    context, authorization = _with_frozen_command_plan(tmp_path, context, authorization)
    ledger = FakeLedger(authorization)
    journal = LedgerCommandJournal(tmp_path, ledger)
    command = journal._command_payload(
        context,
        authorization,
        command_id="command-full-unknown",
        argv=("python", "producer.py"),
        cwd="work",
        source_snapshot_hash="f" * 64,
        expected_outputs=(),
    )
    ledger.start_phase_command(command)
    calls: list[str] = []

    outcome = journal.run_once(
        context,
        command_id="command-full-unknown",
        argv=("python", "producer.py"),
        cwd="work",
        source_snapshot_hash="f" * 64,
        expected_outputs=(),
        runner=lambda: calls.append("run") or _command_result(tmp_path),
    )

    assert outcome["status"] == "unknown"
    assert calls == []


def test_journal_rejects_wrong_source_before_start(tmp_path: Path) -> None:
    context, authorization = _command_context(tmp_path)
    context, authorization = _with_frozen_command_plan(tmp_path, context, authorization)
    ledger = FakeLedger(authorization)
    journal = LedgerCommandJournal(tmp_path, ledger)

    with pytest.raises(CommandJournalError, match="source snapshot"):
        journal.run_once(
            context,
            command_id="command-full-wrong-source",
            argv=("python", "producer.py"),
            cwd="work",
            source_snapshot_hash="e" * 64,
            expected_outputs=(),
            runner=lambda: _command_result(tmp_path),
        )

    assert ledger.commands == {}


def test_executor_capability_exists_only_inside_authorized_callback(tmp_path: Path) -> None:
    authorization = replace(_authorization(phase="full"), adapter_identity="adapter-generic")
    context = replace(
        _context(tmp_path, phase="full"),
        adapter_identity=authorization.adapter_identity,
        authorization_hash=authorization.authorization_hash,
    )
    context, authorization = _with_frozen_command_plan(tmp_path, context, authorization)
    authority = Authority(authorization)
    observed = []

    def runner(current):
        capability = require_executor_capability(current)
        observed.append((capability.executor_name, capability.phase_execution_id))
        return _inventory(current)

    GenericExternalPhaseExecutor(authority, runner).execute(context)
    assert observed == [("generic_external", context.phase_execution_id)]
    with pytest.raises(TypedPhaseFailure, match="(?i)executor capability"):
        require_executor_capability(context)

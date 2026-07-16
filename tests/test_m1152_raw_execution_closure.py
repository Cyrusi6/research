from __future__ import annotations

import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from auto_research.adapters.runner import ExperimentRunner
from auto_research.agents.experiment import ExperimentAgent
from auto_research.command_journal import LedgerCommandJournal
from auto_research.contract_store import ContractStore
from auto_research.domain_contracts import validate_trial_spec
from auto_research.failure_validation import canonical_evidence_bytes, evidence_bytes_hash
from auto_research.research_state import IntegrityError, ResearchEventLedger
from support.authoritative_evidence import build_failure_evidence_v4
from test_m115_failure_resume_reducer import _full_running_proxy_attempt, _resource_pause, _resume
from test_m115_final_e2e import _generic_project


def test_runner_executes_frozen_argv_and_environment_without_shell_roundtrip(tmp_path: Path) -> None:
    script = tmp_path / "producer.py"
    script.write_text(
        "import os,sys\n"
        "assert os.environ['PYTHONPATH'].split(os.pathsep)[0] == sys.argv[1]\n"
        "print('typed-argv-ok')\n",
        encoding="utf-8",
    )
    inherited = os.environ.get("PYTHONPATH", "")
    result = ExperimentRunner({}).run_step(
        name="typed-argv",
        argv=[sys.executable, str(script), str(tmp_path)],
        working_dir=tmp_path,
        environment={"PYTHONPATH": os.pathsep.join(item for item in (str(tmp_path), inherited) if item)},
        inherited_environment=("PATH",),
    )
    assert result["status"] == "ok"
    assert result["returncode"] == 0
    assert result["attempts"][0]["argv"] == [sys.executable, str(script), str(tmp_path)]


def test_real_sample_manifest_recomputes_ordered_ids_from_raw_sample_bytes(tmp_path: Path) -> None:
    root = tmp_path / "sample-provenance"
    root.mkdir()
    _, _, variant = _generic_project(root)
    trial_spec = json.loads((root / "plan/trial_spec.json").read_text(encoding="utf-8"))
    validate_trial_spec(trial_spec)
    attacked = deepcopy(trial_spec)
    attacked["sample_manifest"]["datasets"][0]["ordered_sample_ids"][0] = "f" * 64
    with pytest.raises(ValueError, match="sample|ordered|content"):
        validate_trial_spec(attacked)


def test_failure_log_hash_must_equal_authoritative_phase_receipt_stderr(tmp_path: Path) -> None:
    ledger, running = _full_running_proxy_attempt(tmp_path)
    failure = build_failure_evidence_v4(
        tmp_path,
        running,
        failure_class="activation_failure",
        suffix="stderr-mismatch",
        exit_code=3,
    )
    failure["log_hash"] = "f" * 64
    failure_raw = canonical_evidence_bytes(failure)
    failure_hash = evidence_bytes_hash(failure_raw)
    failure_path = (
        tmp_path
        / "experiment"
        / "attempts"
        / running["attempt_id"]
        / failure["producer_run_id"]
        / "failure_evidence"
        / f"{failure_hash}.json"
    )
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_bytes(failure_raw)
    before_events = ledger.events()
    before_state = ledger.state()
    with pytest.raises(IntegrityError, match="stderr|log_hash|receipt"):
        ledger.disposition_failure(failure)
    assert ledger.events() == before_events
    assert ledger.state() == before_state


def test_manual_core_resource_receipt_cannot_authorize_resume(tmp_path: Path) -> None:
    ledger, running = _full_running_proxy_attempt(tmp_path)
    failure, _, failure_hash = _resource_pause(tmp_path, running)
    paused, _ = ledger.disposition_failure(failure)
    pause_event = next(event for event in reversed(ledger.events()) if event["event_type"] == "AttemptDispositioned")
    forged = _resume(tmp_path, paused, pause_event, failure, failure_hash)
    before_events = ledger.events()
    before_state = ledger.state()
    with pytest.raises(IntegrityError, match="probe|receipt|command|authority|resume_resource_attempt"):
        ledger.resume_attempt(forged)
    assert ledger.events() == before_events
    assert ledger.state() == before_state


def test_started_without_receipt_restart_returns_authoritative_integrity_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "started-no-receipt"
    root.mkdir()
    context, _, _ = _generic_project(root)
    first = ExperimentAgent(context)
    calls = 0

    def crash(**kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        raise RuntimeError("process disappeared after PhaseCommandStarted")

    monkeypatch.setattr(first.runner, "run_step", crash)
    with pytest.raises(RuntimeError, match="disappeared"):
        first.run()
    assert calls == 1

    restarted = ExperimentAgent(context)
    monkeypatch.setattr(
        restarted.runner,
        "run_step",
        lambda **kwargs: pytest.fail(f"unknown command was rerun: {kwargs}"),
    )
    result = restarted.run()
    assert calls == 1
    assert result["route_outcome"]["next_action"] == "BLOCK_INTEGRITY"
    state = ResearchEventLedger(root).state()
    assert next(iter(state["phase_commands"].values()))["status"] == "unknown"


def test_finalized_attempt_restart_returns_historical_result_without_duplicate_variant(tmp_path: Path) -> None:
    root = tmp_path / "finalized-replay"
    root.mkdir()
    context, _, _ = _generic_project(root)
    first = ExperimentAgent(context).run()
    before = ResearchEventLedger(root).state()
    replay = ExperimentAgent(context).run()
    after = ResearchEventLedger(root).state()
    assert replay["attempt"]["attempt_id"] == first["attempt"]["attempt_id"]
    assert replay["route_outcome"] == first["route_outcome"]
    assert after["last_sequence"] == before["last_sequence"]
    assert after["trial_results"] == before["trial_results"]


def test_phase_receipt_validator_rejects_command_identity_and_log_ref_drift(tmp_path: Path) -> None:
    root = tmp_path / "receipt-drift"
    root.mkdir()
    context, _, _ = _generic_project(root)
    ExperimentAgent(context).run()
    ledger = ResearchEventLedger(root)
    command_id, record = next(iter(ledger.state()["phase_commands"].items()))
    receipt = ContractStore(root).read_json(record["receipt_ref"], schema_file="phase_run_receipt_v4.schema.json")
    baseline_events = ledger.events()
    for field, value in (
        ("command_spec_id", "forged-command-spec"),
        ("command_plan_hash", "e" * 64),
        ("stdout_hash", "d" * 64),
        ("stderr_hash", "c" * 64),
    ):
        attacked = deepcopy(receipt)
        attacked[field] = value
        ref = ContractStore(root).put_json(attacked, schema_file="phase_run_receipt_v4.schema.json")
        with pytest.raises(IntegrityError, match="PhaseRunReceipt|receipt|command"):
            ledger.complete_phase_command(command_id, ref)
        assert ledger.events() == baseline_events


def test_c2c_no_gpu_must_return_pause_route_not_quarantine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from support.local_c2c_execution import build_c2c_context, create_local_c2c_repo

    repo = create_local_c2c_repo(tmp_path / "fixture", proxy_accuracy=0.51)
    root = tmp_path / "c2c-no-gpu"
    root.mkdir()
    context = build_c2c_context(root, repo, profile="standard")
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    result = ExperimentAgent(context).run()
    assert result["route_outcome"]["next_action"] == "PAUSE_RESOURCE"
    attempt = result["attempt"]
    assert attempt["state"] == "RESOURCE_PAUSED"
    assert ResearchEventLedger(root).state()["directions"][attempt["direction_semantic_hash"]]["budget"] == {
        "target": 5,
        "reserved": 1,
        "consumed": 0,
    }


def test_readiness_blocked_cannot_be_overridden_by_zero_exit_activation(tmp_path: Path) -> None:
    validator = getattr(__import__("auto_research.proxy_classifier", fromlist=["derive_readiness_from_receipts"]), "derive_readiness_from_receipts", None)
    assert callable(validator), "production readiness must be derived from raw command receipts"
    outcome = validator(
        required_check_ids=("activation", "full-ready"),
        raw_checks={
            "activation": {"status": "PASS", "exit_code": 0},
            "full-ready": {"status": "BLOCKED", "ready": False, "exit_code": 0},
        },
    )
    assert outcome["ready"] is False
    assert outcome["checks"][1]["status"] != "PASS"

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

from auto_research.agents.experiment import ExperimentAgent
from auto_research.proxy_classifier import derive_readiness_from_receipts
from auto_research.research_state import ResearchEventLedger
from support.local_c2c_execution import build_c2c_context, create_local_c2c_repo, install_fake_gpu


def test_c2c_command_plan_uses_core_derivation_not_virtual_collect(tmp_path: Path) -> None:
    agent = ExperimentAgent.__new__(ExperimentAgent)
    agent.context = SimpleNamespace(project_root=tmp_path)

    values = agent._c2c_command_spec_values(
        [("proxy_command_1", {"argv": ["python", "producer.py"]})],
        phase="proxy",
        cwd=tmp_path,
    )

    assert [item["command_spec_id"] for item in values] == [
        "proxy-proxy_command_1",
        "proxy-derive-evidence",
    ]
    assert values[-1]["dependencies"] == ["proxy-proxy_command_1"]
    source = inspect.getsource(ExperimentAgent._run_authoritative_step)
    assert "internal_output_command" not in source
    assert "collect-evidence" not in inspect.getsource(ExperimentAgent._execute_c2c_adapter_phase)


def test_receipt_authorized_readiness_block_is_not_overridden_by_activation_exit_zero(
    tmp_path: Path,
) -> None:
    del tmp_path
    readiness = derive_readiness_from_receipts(
        required_check_ids=("activation", "full-ready"),
        raw_checks={
            "activation": {"status": "PASS", "exit_code": 0},
            "full-ready": {"status": "BLOCKED", "ready": False, "exit_code": 0},
        },
    )
    assert readiness["ready"] is False
    assert [item["status"] for item in readiness["checks"]] == ["PASS", "BLOCKED"]


def test_non_simulated_c2c_phase_commits_core_derivation_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    install_fake_gpu(tmp_path, monkeypatch)
    repo = create_local_c2c_repo(tmp_path / "fixture", proxy_accuracy=0.51)
    root = tmp_path / "c2c-derivation"
    root.mkdir()
    agent = ExperimentAgent(build_c2c_context(root, repo, profile="standard"))

    result = agent.run()

    assert result.get("attempt", {}).get("state") == "METHOD_COMPLETED", result
    events = ResearchEventLedger(root).events()
    command_events = [
        event
        for event in events
        if event["event_type"] == "PhaseCommandStarted"
    ]
    specs = [event["payload"]["command"]["command_spec_id"] for event in command_events]
    assert "proxy-derive-evidence" in specs
    assert "full-derive-evidence" in specs
    assert not any("collect-evidence" in spec for spec in specs)
    phase_commands = ResearchEventLedger(root).state()["phase_commands"]
    derivations = [
        record
        for record in phase_commands.values()
        if record["command"]["command_spec_id"].endswith("-derive-evidence")
    ]
    assert len(derivations) == 2
    assert all(record["status"] == "completed" and record["receipt_ref"] for record in derivations)

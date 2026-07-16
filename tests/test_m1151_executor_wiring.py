from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from auto_research.agents import experiment as experiment_module
from auto_research.agents.experiment import ExperimentAgent
from auto_research.phase_execution import (
    C2CFullPhaseExecutor,
    C2CProxyPhaseExecutor,
    GenericExternalPhaseExecutor,
    SyntheticPhaseExecutor,
)
from auto_research.research_state import IntegrityError
from auto_research.utils import read_json
from support.local_c2c_execution import build_c2c_context, create_local_c2c_repo, install_fake_gpu
from test_m115_final_e2e import _generic_project


@pytest.fixture(autouse=True)
def _hermetic_c2c_dataset_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    hf_home = tmp_path_factory.mktemp("m1151-hf-home")
    dataset_cache = hf_home / "datasets"
    for dataset_dir in ("mmlu-redux", "ai2-arc", "openbookqa"):
        (dataset_cache / dataset_dir).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(dataset_cache))


def _install_executor_spy(
    monkeypatch: pytest.MonkeyPatch,
    executor_name: str,
    executor_type: type,
    calls: list[tuple[str, str]],
) -> None:
    class SpyExecutor:
        def __init__(self, *args, **kwargs):
            self._delegate = executor_type(*args, **kwargs)

        def execute(self, context):
            calls.append((executor_name, context.phase))
            return self._delegate.execute(context)

        __call__ = execute

    monkeypatch.setattr(experiment_module, executor_name, SpyExecutor, raising=False)


def test_experiment_run_structurally_owns_all_phase_executors() -> None:
    source = inspect.getsource(ExperimentAgent.run)

    for executor_name in (
        "C2CProxyPhaseExecutor",
        "C2CFullPhaseExecutor",
        "GenericExternalPhaseExecutor",
        "SyntheticPhaseExecutor",
    ):
        assert hasattr(experiment_module, executor_name), f"ExperimentAgent does not import {executor_name}"
        assert executor_name in source, f"ExperimentAgent.run does not instantiate {executor_name}"

    for direct_entry in (
        "self._run_c2c_small_loop(",
        "self._run_generic_external_phase(",
        "self._run_simulated(",
    ):
        assert direct_entry not in source, f"ExperimentAgent.run still bypasses PhaseExecutor via {direct_entry}"


def test_generic_non_simulated_run_uses_generic_external_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "generic"
    root.mkdir()
    context, _, _ = _generic_project(root)
    calls: list[tuple[str, str]] = []
    _install_executor_spy(
        monkeypatch,
        "GenericExternalPhaseExecutor",
        GenericExternalPhaseExecutor,
        calls,
    )

    ExperimentAgent(context).run()

    assert calls == [("GenericExternalPhaseExecutor", "full")]


def test_c2c_standard_run_uses_proxy_then_full_executors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_gpu(tmp_path, monkeypatch)
    repo = create_local_c2c_repo(tmp_path / "fixture-c2c", proxy_accuracy=0.51)
    root = tmp_path / "c2c"
    root.mkdir()
    context = build_c2c_context(root, repo, profile="standard")
    calls: list[tuple[str, str]] = []
    _install_executor_spy(monkeypatch, "C2CProxyPhaseExecutor", C2CProxyPhaseExecutor, calls)
    _install_executor_spy(monkeypatch, "C2CFullPhaseExecutor", C2CFullPhaseExecutor, calls)
    agent = ExperimentAgent(context)

    result = agent.run()

    assert calls == [
        ("C2CProxyPhaseExecutor", "proxy"),
        ("C2CFullPhaseExecutor", "full"),
    ], result
    trial_spec = read_json(root / "plan/trial_spec.json", default={}) or {}
    for phase_contract in trial_spec["phase_contracts"]:
        commands = phase_contract["command_plan"]["commands"]
        assert commands
        assert all("freeze-required" not in command["argv"] for command in commands)
        assert [command["ordinal"] for command in commands] == list(range(len(commands)))
        assert commands[0]["dependencies"] == []
        positions = {command["command_spec_id"]: index for index, command in enumerate(commands)}
        assert all(
            dependency in positions and positions[dependency] < index
            for index, command in enumerate(commands)
            for dependency in command["dependencies"]
        )
        unconditional = [command for command in commands if command["condition"]["kind"] == "always"]
        assert all(command["dependencies"] for command in unconditional[1:])
        recovery_commands = [command for command in commands if "train_recovery" in command["command_spec_id"]]
        assert all(command["condition"]["kind"] != "always" for command in recovery_commands)
        assert all("full-train" in command["dependencies"] for command in recovery_commands)
    command_events = [
        (event["sequence"], event["payload"]["command"]["command_spec_id"])
        for event in experiment_module.ResearchEventLedger(root).events()
        if event["event_type"] == "PhaseCommandStarted"
    ]
    command_sequences = {command_spec_id: sequence for sequence, command_spec_id in command_events}
    assert command_sequences["proxy-derive-evidence"] < command_sequences["full-train"]


def test_c2c_proxy_reject_never_invokes_full_executor_or_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_gpu(tmp_path, monkeypatch)
    repo = create_local_c2c_repo(tmp_path / "fixture-c2c-reject", proxy_accuracy=0.49)
    root = tmp_path / "c2c-reject"
    root.mkdir()
    context = build_c2c_context(root, repo, profile="standard")
    calls: list[tuple[str, str]] = []
    _install_executor_spy(monkeypatch, "C2CProxyPhaseExecutor", C2CProxyPhaseExecutor, calls)
    _install_executor_spy(monkeypatch, "C2CFullPhaseExecutor", C2CFullPhaseExecutor, calls)
    agent = ExperimentAgent(context)

    result = agent.run()

    assert calls == [("C2CProxyPhaseExecutor", "proxy")]
    assert result.get("route_outcome", {}).get("next_action") == "PROPOSE_NEXT_VARIANT", result
    assert "FullPhaseStarted" not in [
        event["event_type"] for event in experiment_module.ResearchEventLedger(root).events()
    ]


def test_simulated_run_uses_synthetic_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "synthetic"
    root.mkdir()
    context, _, _ = _generic_project(root)
    context.config["experiment"]["simulate"] = True
    calls: list[tuple[str, str]] = []
    _install_executor_spy(monkeypatch, "SyntheticPhaseExecutor", SyntheticPhaseExecutor, calls)

    ExperimentAgent(context).run()

    assert calls == [("SyntheticPhaseExecutor", "full")]


def test_direct_authoritative_step_cannot_bypass_phase_executor_before_runner(tmp_path: Path) -> None:
    agent = object.__new__(ExperimentAgent)
    agent.context = SimpleNamespace(project_root=tmp_path)
    agent._active_phase_context = SimpleNamespace(
        phase="full",
        phase_execution_id="phase-full-direct-bypass",
        implementation_hash="a" * 64,
    )
    runner_calls: list[str] = []

    class Runner:
        def run_step(self, **kwargs):
            runner_calls.append(str(kwargs.get("name")))
            return {"status": "ok", "returncode": 0, "stdout": "", "stderr": ""}

    class Journal:
        def run_once(self, context, *, runner, **kwargs):
            del context, kwargs
            runner()
            return {"status": "completed"}

    agent.runner = Runner()
    agent._active_command_journal = Journal()

    with pytest.raises(IntegrityError, match="PhaseExecutor|executor authorization|executor capability"):
        agent._run_authoritative_step(
            name="direct-bypass",
            command={"argv": ["true"], "cwd": str(tmp_path)},
            cwd=tmp_path,
        )

    assert runner_calls == []


@pytest.mark.parametrize(
    ("entrypoint", "expected_phase", "kwargs"),
    [
        (
            "_run_single_c2c_proxy_candidate",
            "proxy",
            {
                "adapter": None,
                "candidate": {},
                "index": 0,
                "simulate": False,
                "baseline_mean": 0.0,
                "min_delta": 0.0,
                "max_regression": 0.0,
                "gpu_selection": None,
                "proxy_gpu_selection": None,
            },
        ),
        (
            "_run_single_c2c_full_candidate",
            "full",
            {
                "adapter": None,
                "candidate": {},
                "index": 0,
                "simulate": False,
                "baseline_mean": 0.0,
                "min_delta": 0.0,
                "max_regression": 0.0,
                "gpu_selection": None,
                "proxy_gpu_selection": None,
            },
        ),
    ],
)
def test_direct_c2c_side_effect_callbacks_fail_before_filesystem_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    expected_phase: str,
    kwargs: dict,
) -> None:
    agent = object.__new__(ExperimentAgent)
    agent.context = SimpleNamespace(project_root=tmp_path, config={})
    agent._active_phase_context = None
    agent._active_command_journal = None
    side_effects: list[str] = []
    monkeypatch.setattr(agent, "_load_frozen_c2c_patch", lambda *args, **values: side_effects.append("patch") or {})

    with pytest.raises(IntegrityError, match="executor-owned|executor capability"):
        getattr(agent, entrypoint)(**kwargs)

    assert side_effects == [], f"{expected_phase} callback performed work before executor authorization"


def test_direct_synthetic_callback_fails_before_artifact_write(tmp_path: Path) -> None:
    agent = object.__new__(ExperimentAgent)
    agent.context = SimpleNamespace(project_root=tmp_path)
    agent._active_phase_context = None
    agent._active_command_journal = None

    with pytest.raises(IntegrityError, match="executor-owned|executor capability"):
        agent._run_simulated(
            {},
            "env.md",
            None,
            attempt={"attempt_kind": "full"},
            trial_spec={},
        )

    assert list(tmp_path.iterdir()) == []

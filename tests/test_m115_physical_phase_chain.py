from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from auto_research.agents.experiment import (
    ExperimentAgent,
    _c2c_strict_evidence_inventory,
    _decode_staged_execution_observations,
    _stage_evidence_inventory,
)
import auto_research.agents.experiment as experiment_module
from auto_research.command_journal import LedgerCommandJournal
from auto_research.phase_execution import ResearchLedgerPhaseAuthority
from auto_research.research_state import ResearchEventLedger
from test_m114_authoritative_phase_transactions import _c2c_inputs


def _proxy_comparison(*, baseline_value: float, proxy_value: float, full_value: float) -> dict:
    return {
        "metrics": {"mean": full_value, "datasets": {"fake": full_value}},
        "proxy_screen": {
            "status": "passed",
            "metrics": {"mean": proxy_value, "datasets": {"fake": proxy_value}},
            "baseline_metrics": {"mean": baseline_value, "datasets": {"fake": baseline_value}},
            "proxy_baseline": {"mean": baseline_value, "datasets": {"fake": baseline_value}},
        },
        "ablation": {"metrics": {"datasets": {"fake": full_value - 0.1}}},
        "matched_control_metrics": {"datasets": {"fake": full_value - 0.2}},
        "coverage_metrics": {"datasets": {"fake": 1.0}},
    }


def test_proxy_inventory_uses_proxy_command_metrics_not_full_aggregate(tmp_path: Path) -> None:
    attempt, trial_spec, _, baseline = _c2c_inputs(tmp_path)
    comparison = _proxy_comparison(baseline_value=10.0, proxy_value=11.0, full_value=99.0)

    inventory = _c2c_strict_evidence_inventory(
        project_root=tmp_path,
        attempt=attempt,
        trial_spec=trial_spec,
        comparison_candidate=comparison,
        baseline=baseline,
        simulate=False,
    )
    completion = _stage_evidence_inventory(
        project_root=tmp_path,
        attempt=attempt,
        trial_spec=trial_spec,
        inventory=inventory,
    )
    rows = _decode_staged_execution_observations(tmp_path, attempt, trial_spec, completion)

    assert {(row["role"], row["metric_value"]) for row in rows} == {("baseline", 10.0), ("candidate", 11.0)}
    assert all(row["metric_value"] != 99.0 for row in rows)


def test_full_callbacks_observe_committed_proxy_and_full_phase_started(tmp_path: Path, monkeypatch) -> None:
    attempt, trial_spec, _, baseline = _c2c_inputs(tmp_path)
    ledger = ResearchEventLedger(tmp_path)
    comparison = _proxy_comparison(baseline_value=0.0, proxy_value=1.0, full_value=9.0)
    inventory = _c2c_strict_evidence_inventory(
        project_root=tmp_path,
        attempt=attempt,
        trial_spec=trial_spec,
        comparison_candidate=comparison,
        baseline=baseline,
        simulate=False,
    )
    completion = _stage_evidence_inventory(
        project_root=tmp_path,
        attempt=attempt,
        trial_spec=trial_spec,
        inventory=inventory,
    )
    proxy_attempt, route = ledger.commit_proxy_evidence(completion)
    assert route["next_action"] == "RUN_FULL"
    full_attempt = ledger.start_full_phase(
        proxy_attempt["attempt_id"],
        phase_execution_id="phase-full-authorized",
        producer_run_id="producer-full-authorized",
    )

    callback_states: list[tuple[str, tuple[str, ...]]] = []

    class Runner:
        def run_step(self, **kwargs):
            state = ledger.state()
            current = state["attempts"][attempt["attempt_id"]]
            event_types = tuple(event["event_type"] for event in ledger.events())
            callback_states.append((current["state"], event_types))
            return {"status": "ok", "name": kwargs["name"], "returncode": 0}

    class Adapter:
        repo_root = tmp_path
        baseline = {"mean": 0.0, "datasets": {"fake": 0.0}}

        def materialize_candidate_configs(self, candidate, gpu_selection, *, proxy_gpu_selection):
            return {
                "run_id": "candidate-full-run",
                "run_root": tmp_path / "candidate-full-run",
                "run_state_path": tmp_path / "candidate-full-run" / "run_state.json",
                "preflight_path": tmp_path / "candidate-full-run" / "preflight.json",
                "commands": {"preflight": [], "train": "train", "eval": ["eval"]},
                "eval_configs": {"fake": tmp_path / "eval.yaml"},
                "frozen_hashes": {},
                "has_executable_change": True,
            }

        def preflight(self, run_spec, gpu_selection):
            return {"status": "ok", "recovery_actions": []}

        def collect_candidate_metrics(self, run_id):
            return {"mean": 1.0, "datasets": {"fake": 1.0}}

    agent = object.__new__(ExperimentAgent)
    monkeypatch.setattr(experiment_module, "_c2c_execution_repo_path_audit", lambda **kwargs: {"status": "ok"})
    monkeypatch.setattr(experiment_module, "_c2c_execution_repo_output_audit", lambda **kwargs: {"status": "ok"})
    agent.context = SimpleNamespace(
        project_root=tmp_path,
        config={"experiment": {"retry": {}}, "c2c": {"small_loop": {}}},
    )
    agent.runner = Runner()
    phase_authority = ResearchLedgerPhaseAuthority(ledger)
    agent._active_phase_context = phase_authority.context_for_attempt(
        tmp_path,
        full_attempt["attempt_id"],
        "full",
    )
    agent._active_command_journal = LedgerCommandJournal(tmp_path, ledger)
    agent._load_frozen_c2c_patch = lambda candidate: {}
    agent._prepare_c2c_execution_repo = lambda candidate, adapter, patch: {"status": "skipped"}
    agent._apply_frozen_c2c_patch = lambda candidate, adapter, patch, execution_repo: {"status": "skipped", "changed_files": ["candidate.py"]}
    agent._load_reusable_c2c_proxy_state = lambda run_spec, patch_fingerprint: None
    agent._run_c2c_ablation_eval = lambda **kwargs: {"enabled": False, "status": "skipped", "attempts": []}

    result = agent._run_single_c2c_full_candidate(
        adapter=Adapter(),
        candidate={"id": "candidate", "title": "candidate", "hypothesis": "test"},
        index=0,
        simulate=False,
        baseline_mean=0.0,
        min_delta=0.1,
        max_regression=1.0,
        gpu_selection=SimpleNamespace(selected_ids=[]),
        proxy_gpu_selection=SimpleNamespace(selected_ids=[]),
    )

    assert full_attempt["state"] == "FULL_RUNNING"
    assert result["metrics"]["datasets"] == {"fake": 1.0}
    assert callback_states
    for state, event_types in callback_states:
        assert state == "FULL_RUNNING"
        assert event_types.index("ProxyEvidenceCommitted") < event_types.index("FullPhaseStarted")

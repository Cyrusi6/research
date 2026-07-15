from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from auto_research.agents.experiment import (
    ExperimentAgent,
    _c2c_strict_evidence_inventory,
    _stage_evidence_inventory,
)
import auto_research.agents.experiment as experiment_module
from auto_research.command_journal import LedgerCommandJournal
from auto_research.domain_contracts import canonical_hash
from auto_research.phase_execution import (
    C2CFullPhaseExecutor,
    PhaseArtifactInventory,
    ResearchLedgerPhaseAuthority,
)
from auto_research.research_state import ResearchEventLedger
from support.authoritative_evidence import (
    record_completed_evidence_command,
    validate_authoritative_completion,
)
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
        "activation_smoke": {"status": "passed", "attempts": [{"status": "ok"}], "implementation_surface_ids": ["src/router.py"]},
        "full_s3_readiness": {"status": "ready", "full_train_allowed": True},
        "ablation": {"metrics": {"datasets": {"fake": full_value - 0.1}}},
        "matched_control_metrics": {"datasets": {"fake": full_value - 0.2}},
        "coverage_metrics": {"datasets": {"fake": 1.0}},
    }


def test_proxy_inventory_uses_proxy_command_metrics_not_full_aggregate(tmp_path: Path) -> None:
    attempt, trial_spec, _, baseline = _c2c_inputs(tmp_path)
    ledger = ResearchEventLedger(tmp_path)
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
    record_completed_evidence_command(tmp_path, ledger, attempt, completion)
    rows = validate_authoritative_completion(
        tmp_path,
        ledger,
        attempt,
        completion,
    ).observations

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
    record_completed_evidence_command(tmp_path, ledger, attempt, completion)
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
    phase_context = phase_authority.context_for_attempt(
        tmp_path,
        full_attempt["attempt_id"],
        "full",
    )
    agent._active_phase_context = phase_context
    agent._active_command_journal = LedgerCommandJournal(tmp_path, ledger)
    agent._load_frozen_c2c_patch = lambda candidate: {}
    agent._prepare_c2c_execution_repo = lambda candidate, adapter, patch: {"status": "skipped"}
    agent._apply_frozen_c2c_patch = lambda candidate, adapter, patch, execution_repo: {"status": "skipped", "changed_files": ["candidate.py"]}
    agent._run_c2c_ablation_eval = lambda **kwargs: {"enabled": False, "status": "skipped", "attempts": []}

    result_holder: dict[str, dict] = {}

    def execute_full(context):
        raw_output = tmp_path / "experiment" / "phase-authority" / "main_results.json"
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        raw_output.write_text("{}", encoding="utf-8")
        command_result = agent._run_authoritative_step(
            name="command_0",
            command=["fixture-phase-command", "full"],
            command_spec_id="full-command-000",
            working_dir=tmp_path,
            authoritative_output_factory=lambda: [
                {
                    "kind": "main_results",
                    "source_path": raw_output.relative_to(tmp_path).as_posix(),
                }
            ],
        )
        assert command_result["status"] == "ok"
        result_holder["result"] = {"metrics": {"datasets": {"fake": 1.0}}}
        artifacts = []
        for kind in context.expected_evidence_kinds:
            relative_path = f"experiment/phase-authority/{kind}.json"
            path = tmp_path / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            artifacts.append({
                "kind": kind,
                "source_path": relative_path,
                "content_hash": canonical_hash({}),
                "receipt_hash": canonical_hash({"kind": kind, "phase": context.phase_execution_id}),
                "producer_run_id": context.producer_run_id,
            })
        return PhaseArtifactInventory(context=context, artifacts=artifacts)

    C2CFullPhaseExecutor(phase_authority, execute_full).execute(phase_context)
    result = result_holder["result"]

    assert full_attempt["state"] == "FULL_RUNNING"
    assert result["metrics"]["datasets"] == {"fake": 1.0}
    assert callback_states
    for state, event_types in callback_states:
        assert state == "FULL_RUNNING"
        assert event_types.index("ProxyEvidenceCommitted") < event_types.index("FullPhaseStarted")

import json
from pathlib import Path
from types import SimpleNamespace

import yaml
import pytest

import auto_research.config as config_module
import auto_research.agents.literature as literature_module
import auto_research.orchestrator as orchestrator_module
from auto_research.adapters.literature import LiteratureProvider
from auto_research.orchestrator import Orchestrator
from auto_research.registry import block_stage, default_registry
from auto_research.s2_feedback_policy import build_s2_adaptive_policy, build_s2_feedback_context, build_s2_score_adjustment_report
from auto_research.s2_planner_contracts import build_s2_candidate_pool, build_s2_planner_gate_report, build_s2_variant_scorecard
from auto_research.utils import write_json, write_yaml


def _test_config(tmp_path: Path, simulate: bool) -> dict:
    return {
        "project": {"workspace_root": str(tmp_path), "target_venue": "TestConf", "language": "en"},
        "llm": {"use_real_api": False, "model": "mock"},
        "literature": {"download_pdfs": True, "request_timeout_seconds": 1, "max_papers": 2, "arxiv_max_results": 1},
        "plan": {"min_hypotheses": 1, "min_baselines": 2, "min_datasets": 1},
        "experiment": {"simulate": simulate, "random_seeds": [42, 123, 456]},
        "writing": {"claim_verification": {"enabled": True, "min_pass_rate": 0.8}, "require_compile": False},
        "review": {"pass_threshold": 7.0, "max_iterations": 2},
        "orchestration": {"judge_max_retries": 1},
    }


def _generic_s1_codex_payload() -> dict:
    return {
        "schema_version": "generic_s1_codex_direction_v1",
        "status": "ok",
        "evidence_requests": [
            {
                "query": "retrieval benchmark method direction",
                "source_type": "paper",
                "desired_evidence": "method",
                "why_needed": "S2 needs one high-level idea to plan.",
            }
        ],
        "evidence_bundle": {
            "items": [
                {
                    "source_path": "literature/survey.md",
                    "source_type": "artifact",
                    "summary": "The topic has enough literature context to choose a bounded retrieval benchmark idea.",
                    "supports": ["retrieval_alignment_direction"],
                    "risks": [],
                },
                {
                    "source_path": "literature/survey.md#limitations",
                    "source_type": "artifact",
                    "summary": "Baseline comparability is a falsification risk.",
                    "supports": [],
                    "risks": ["baseline comparability"],
                }
            ]
        },
        "novelty_audits": [
            {
                "direction_id": "retrieval_alignment_direction",
                "status": "pass",
                "enabled": True,
                "passed": True,
                "threshold": 0.58,
                "novelty_score": 0.75,
            }
        ],
        "direction_decision": {
            "direction_id": "retrieval_alignment_direction",
            "title": "Retrieval Alignment Direction",
            "core_hypothesis": "A bounded retrieval alignment intervention can improve the primary benchmark over a baseline.",
            "allowed_variants": ["lightweight alignment loss", "retrieval scoring calibration"],
            "forbidden_patterns": ["unbounded architecture rewrite"],
            "target_datasets": ["TBD-benchmark"],
            "failure_focus": ["baseline comparability", "ablation signal"],
            "rationale": "This is a high-level S1 direction; S2 will turn it into an experiment plan.",
        },
        "selected_ideas": [
            {
                "id": "retrieval_alignment_direction",
                "title": "Retrieval Alignment Direction",
                "selected": True,
                "hypothesis": "A bounded retrieval alignment intervention can improve the primary benchmark over a baseline.",
                "novelty_score": 7,
                "feasibility_score": 7,
                "description": "High-level direction only, not a concrete S2 variant.",
                "motivation": "The project needs one focused direction before experiment planning.",
                "expected_contribution": "Improved retrieval benchmark effectiveness with an ablation path.",
                "key_baselines": ["Strong retrieval baseline", "Ablation-off control"],
                "required_compute": "1-4 GPU days",
                "key_references": ["literature/survey.md"],
                "evidence_refs": [{"source_type": "artifact", "source_label": "literature/survey.md", "claim": "survey supports direction"}],
                "counterevidence_refs": [{"source_type": "artifact", "source_label": "risk", "claim": "must keep baseline comparable"}],
            }
        ],
        "negative_constraints": {
            "forbidden_idea_ids": [],
            "forbidden_patterns": ["unbounded architecture rewrite"],
            "failure_feedback_rules": ["Use method-level failures only."],
        },
        "decision_chain": {
            "evidence": ["literature/survey.md"],
            "counterevidence": ["baseline comparability"],
            "conclusion": "Use the retrieval alignment direction and let S2 create concrete variants.",
        },
    }


def _mock_generic_s1_codex(monkeypatch):
    monkeypatch.setattr(literature_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    original_subprocess_run = literature_module.subprocess.run

    def fake_run(command, **kwargs):
        if not command or Path(command[0]).name != "codex":
            return original_subprocess_run(command, **kwargs)
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(json.dumps(_generic_s1_codex_payload()), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout='{"type":"thread.started","thread_id":"223e4567-e89b-12d3-a456-426614174001"}\n', stderr="")

    monkeypatch.setattr(literature_module.subprocess, "run", fake_run)


def test_simulated_pipeline_runs_to_completion(monkeypatch, tmp_path: Path) -> None:
    config = _test_config(tmp_path, simulate=True)
    monkeypatch.setattr(config_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)
    monkeypatch.setattr(LiteratureProvider, "search", lambda self, topic: [])
    monkeypatch.setattr(LiteratureProvider, "download_pdf", lambda self, url: None)
    _mock_generic_s1_codex(monkeypatch)
    plan_calls = []
    experiment_calls = []
    original_plan_run = orchestrator_module.PlanAgent.run
    original_experiment_run = orchestrator_module.ExperimentAgent.run

    def tracked_plan_run(self):
        result = original_plan_run(self)
        plan_calls.append(result)
        return result

    def tracked_experiment_run(self, *args, **kwargs):
        result = original_experiment_run(self, *args, **kwargs)
        experiment_calls.append(result)
        return result

    monkeypatch.setattr(orchestrator_module.PlanAgent, "run", tracked_plan_run)
    monkeypatch.setattr(orchestrator_module.ExperimentAgent, "run", tracked_experiment_run)

    orchestrator = Orchestrator()
    project_id = orchestrator.init_project("retrieval benchmark", project_id="proj_pipeline", simulate=True)
    result = orchestrator.start(project_id)

    assert result["status"] == "completed", result
    review_dispatch = tmp_path / project_id / "review" / "revision_dispatch.yaml"
    assert review_dispatch.exists()
    registry = (tmp_path / project_id / "meta" / "registry.yaml").read_text(encoding="utf-8")
    assert "current_stage: DONE" in registry
    state = json.loads((tmp_path / project_id / "orchestration" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["current_stage"] == "DONE"
    assert state["stages"]["S1_literature"]["last_gate"]["passed"] is True
    assert state["stages"]["S1_literature"]["last_gate"]["status"] == "PASS"
    assert state["stages"]["S1_literature"]["last_gate"]["report_path"] == "literature/gate_report.json"
    assert state["stages"]["S1_literature"]["contract_path"] == "orchestration/stage_contracts/S1_literature.json"
    assert state["stages"]["S2_plan"]["attempts"] == 5
    assert state["stages"]["S3_experiment"]["artifacts"][0]["sha256"]
    gate_report = json.loads((tmp_path / project_id / "literature" / "gate_report.json").read_text(encoding="utf-8"))
    assert gate_report["schema_version"] == "stage_gate_v1"
    assert gate_report["status"] == "PASS"
    literature_manifest = json.loads((tmp_path / project_id / "literature" / "stage_manifest.json").read_text(encoding="utf-8"))
    gate_entries = [item for item in literature_manifest["artifacts"] if item["path"] == "literature/gate_report.json"]
    assert gate_entries
    assert gate_entries[0]["created_by"] == "stage-gate-validator"
    s1_contract = json.loads((tmp_path / project_id / "orchestration" / "stage_contracts" / "S1_literature.json").read_text(encoding="utf-8"))
    assert s1_contract["status"] == "completed"
    assert s1_contract["gate"]["status"] == "PASS"
    assert s1_contract["input_hash"]
    assert s1_contract["output_hash"]
    assert any(item["path"] == "literature/direction.json" and item["exists"] for item in s1_contract["produced_outputs"])
    assert any(item["path"] == "literature/candidate_directions.json" and item["exists"] for item in s1_contract["produced_outputs"])
    assert (tmp_path / project_id / "literature" / "direction.json").exists()
    assert (tmp_path / project_id / "literature" / "direction_scorecard.json").exists()
    assert (tmp_path / project_id / "literature" / "novelty_audit.json").exists()
    assert (tmp_path / project_id / "literature" / "evidence_session.json").exists()
    assert (tmp_path / project_id / "plan" / "planner_decision.json").exists()
    assert (tmp_path / project_id / "plan" / "variant.json").exists()
    assert (tmp_path / project_id / "plan" / "trial_spec.json").exists()
    assert (tmp_path / project_id / "plan" / "variant_fingerprint.json").exists()
    ideas = json.loads((tmp_path / project_id / "literature" / "candidate_directions.json").read_text(encoding="utf-8"))
    assert len(ideas) == 1
    assert ideas[0]["s1_evidence_agent"]["source"] == "codex_resume_evidence_agent"
    s3_contract = json.loads((tmp_path / project_id / "orchestration" / "stage_contracts" / "S3_experiment.json").read_text(encoding="utf-8"))
    assert s3_contract["status"] == "completed"
    assert s3_contract["gate"]["report_path"] == "experiment/gate_report.json"
    references_manifest = json.loads((tmp_path / project_id / "references" / "papers" / "manifest.json").read_text(encoding="utf-8"))
    assert references_manifest["papers"]
    research_state = json.loads((tmp_path / project_id / "meta" / "research_state.json").read_text(encoding="utf-8"))
    standard_history = [item for item in research_state["method_tried_history"] if item.get("consumes_direction_budget")]
    assert len(standard_history) == 5
    assert len({item["variant_semantic_hash"] for item in standard_history}) == 5
    aggregate = research_state["latest_direction_aggregate"]
    assert aggregate["outcomes"] and len(aggregate["outcomes"]) == 5
    assert research_state["directions"][aggregate["direction_semantic_hash"]]["budget"] == {"target": 5, "reserved": 0, "consumed": 5}
    assert len(plan_calls) == 5
    assert len(experiment_calls) == 5
    assert len({item["attempt"]["attempt_id"] for item in experiment_calls}) == 5


def test_orchestrator_rejects_sixth_variant_before_plan_agent(monkeypatch, tmp_path: Path) -> None:
    plan_called = False

    class ClosedBudgetLedger:
        def __init__(self, project_root):
            self.project_root = project_root

        def state(self):
            return {
                "current_direction_semantic_hash": "direction-semantic",
                "directions": {
                    "direction-semantic": {
                        "status": "ACTIVE",
                        "budget": {"target": 5, "reserved": 0, "consumed": 5},
                    }
                },
                "last_route_outcome": {"next_action": "FINISH_DIRECTION"},
            }

    def forbidden_plan(*args, **kwargs):
        nonlocal plan_called
        plan_called = True
        raise AssertionError("PlanAgent must not run after the fifth outcome")

    monkeypatch.setattr(orchestrator_module, "ResearchEventLedger", ClosedBudgetLedger)
    monkeypatch.setattr(orchestrator_module.PlanAgent, "run", forbidden_plan)

    with pytest.raises(orchestrator_module.IntegrityError, match="no capacity"):
        Orchestrator._assert_s2_planning_allowed(tmp_path)

    assert plan_called is False


def test_orchestrator_rejects_closed_direction_before_plan_agent(monkeypatch, tmp_path: Path) -> None:
    class ClosedDirectionLedger:
        def __init__(self, project_root):
            self.project_root = project_root

        def state(self):
            return {
                "current_direction_semantic_hash": None,
                "directions": {},
                "last_route_outcome": {"next_action": "FINISH_DIRECTION"},
            }

    monkeypatch.setattr(orchestrator_module, "ResearchEventLedger", ClosedDirectionLedger)

    with pytest.raises(orchestrator_module.IntegrityError, match="S1 must select a new direction"):
        Orchestrator._assert_s2_planning_allowed(tmp_path)


def test_real_mode_blocks_at_experiment_stage(monkeypatch, tmp_path: Path) -> None:
    config = _test_config(tmp_path, simulate=False)
    monkeypatch.setattr(config_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)
    monkeypatch.setattr(LiteratureProvider, "search", lambda self, topic: [])
    monkeypatch.setattr(LiteratureProvider, "download_pdf", lambda self, url: None)
    monkeypatch.setattr(
        literature_module,
        "_run_s1_novelty_auditor",
        lambda **kwargs: {
            "status": "ok",
            "enabled": True,
            "passed": True,
            "threshold": 0.58,
            "audit": {
                "status": "ok",
                "novelty_score": 0.91,
                "max_similarity_score": 0.09,
                "passed": True,
                "most_similar_sources": [],
                "distinctive_elements": ["new causal mechanism"],
                "repeated_patterns": [],
                "revision_guidance": [],
                "decision": "pass",
            },
        },
    )
    _mock_generic_s1_codex(monkeypatch)

    orchestrator = Orchestrator()
    project_id = orchestrator.init_project("real run topic", project_id="proj_blocked", simulate=False)
    result = orchestrator.start(project_id)

    assert result["status"] == "blocked"
    assert result["stage"] == "S2_plan"
    assert "pre-registered sample_count" in result["reason"]
    state = json.loads((tmp_path / project_id / "orchestration" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "blocked"
    assert state["current_stage"] == "S2_plan"
    assert state["stages"]["S3_experiment"]["status"] == "pending"


def test_c2c_real_run_readiness_blocks_before_s0(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "proj_c2c_readiness_block"
    config = _test_config(tmp_path, simulate=False)
    config["c2c"] = {
        "enabled": True,
        "target_repo": str(tmp_path / "missing_repo"),
        "snapshot_path": "external/c2c_snapshot",
        "ref_paper": str(tmp_path / "missing_paper.md"),
        "ref_rebuttal": str(tmp_path / "missing_rebuttal.md"),
        "env_python": str(tmp_path / "missing_python"),
        "dataset_root": str(tmp_path / "missing_datasets"),
        "small_loop": {"strict_dataset_cache": True},
    }
    config["orchestration"]["c2c_e2e"] = {
        "readiness_gate_enabled": True,
        "artifact_audit_enabled": True,
        "block_real_run_on_readiness_fail": True,
    }
    project_root.mkdir(parents=True)
    (project_root / "meta").mkdir()
    write_yaml(project_root / "meta" / "project_config.yaml", config)
    write_yaml(project_root / "meta" / "registry.yaml", default_registry(project_id=project_root.name, topic="topic", config=config))
    monkeypatch.setattr(config_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)

    result = Orchestrator().start(project_root.name)

    assert result["status"] == "blocked"
    assert result["stage"] == "S0_intake"
    assert "C2C real-run readiness failed" in result["reason"]
    readiness = json.loads((project_root / "meta" / "c2c_e2e_readiness_report.json").read_text(encoding="utf-8"))
    assert readiness["gate"] == "fail"
    manifest = json.loads((project_root / "meta" / "c2c_e2e_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["final_status"] == "blocked"


def test_generic_s1_codex_repairs_invalid_json(monkeypatch, tmp_path: Path) -> None:
    config = _test_config(tmp_path, simulate=True)
    config["agents"] = {"s1_evidence_agent": {"max_json_repairs": 1, "timeout_seconds": 5}}
    monkeypatch.setattr(config_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)
    monkeypatch.setattr(LiteratureProvider, "search", lambda self, topic: [])
    monkeypatch.setattr(LiteratureProvider, "download_pdf", lambda self, url: None)
    monkeypatch.setattr(literature_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    original_subprocess_run = literature_module.subprocess.run
    commands = []
    prompts = []

    def fake_run(command, **kwargs):
        if not command or Path(command[0]).name != "codex":
            return original_subprocess_run(command, **kwargs)
        commands.append(command)
        prompts.append(kwargs.get("input") or "")
        output_path = Path(command[command.index("--output-last-message") + 1])
        if len(commands) == 1:
            output_path.write_text("bad json", encoding="utf-8")
            stdout = '{"type":"thread.started","thread_id":"223e4567-e89b-12d3-a456-426614174002"}\n'
        else:
            output_path.write_text(json.dumps(_generic_s1_codex_payload()), encoding="utf-8")
            stdout = ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(literature_module.subprocess, "run", fake_run)

    project_id = Orchestrator().init_project("retrieval benchmark", project_id="proj_generic_s1_repair", simulate=True)
    result = Orchestrator().start(project_id)
    root = tmp_path / project_id

    assert result["status"] == "completed", result
    session = json.loads((root / "literature" / "evidence_session.json").read_text(encoding="utf-8"))
    assert session["repair_count"] == 1
    assert len(session["attempts"]) == 2
    assert "resume" in commands[1]
    assert "errors_to_fix" in prompts[1]

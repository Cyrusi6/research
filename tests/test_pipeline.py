import json
from pathlib import Path
from types import SimpleNamespace

import yaml

import auto_research.config as config_module
import auto_research.agents.literature as literature_module
import auto_research.orchestrator as orchestrator_module
from auto_research.adapters.literature import LiteratureProvider
from auto_research.orchestrator import Orchestrator
from auto_research.registry import block_stage
from auto_research.utils import write_json


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
                }
            ]
        },
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
        if not command or command[0] != "codex":
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

    orchestrator = Orchestrator()
    project_id = orchestrator.init_project("retrieval benchmark", project_id="proj_pipeline", simulate=True)
    result = orchestrator.start(project_id)

    assert result["status"] == "completed"
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
    assert state["stages"]["S2_plan"]["attempts"] == 1
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
    assert any(item["path"] == "literature/ideas.json" and item["exists"] for item in s1_contract["produced_outputs"])
    assert (tmp_path / project_id / "literature" / "evidence_session.json").exists()
    ideas = json.loads((tmp_path / project_id / "literature" / "ideas.json").read_text(encoding="utf-8"))
    assert len(ideas) == 1
    assert ideas[0]["s1_evidence_agent"]["source"] == "codex_resume_evidence_agent"
    s3_contract = json.loads((tmp_path / project_id / "orchestration" / "stage_contracts" / "S3_experiment.json").read_text(encoding="utf-8"))
    assert s3_contract["status"] == "completed"
    assert s3_contract["gate"]["report_path"] == "experiment/gate_report.json"
    references_manifest = json.loads((tmp_path / project_id / "references" / "papers" / "manifest.json").read_text(encoding="utf-8"))
    assert references_manifest["papers"]


def test_real_mode_blocks_at_experiment_stage(monkeypatch, tmp_path: Path) -> None:
    config = _test_config(tmp_path, simulate=False)
    monkeypatch.setattr(config_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)
    monkeypatch.setattr(LiteratureProvider, "search", lambda self, topic: [])
    monkeypatch.setattr(LiteratureProvider, "download_pdf", lambda self, url: None)
    _mock_generic_s1_codex(monkeypatch)

    orchestrator = Orchestrator()
    project_id = orchestrator.init_project("real run topic", project_id="proj_blocked", simulate=False)
    result = orchestrator.start(project_id)

    assert result["status"] == "blocked"
    assert result["stage"] == "S3_experiment"
    state = json.loads((tmp_path / project_id / "orchestration" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "blocked"
    assert state["current_stage"] == "S3_experiment"
    assert state["stages"]["S3_experiment"]["last_error"]
    s3_contract = json.loads((tmp_path / project_id / "orchestration" / "stage_contracts" / "S3_experiment.json").read_text(encoding="utf-8"))
    assert s3_contract["status"] == "blocked"
    assert s3_contract["reason"]


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
        if not command or command[0] != "codex":
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

    assert result["status"] == "completed"
    session = json.loads((root / "literature" / "evidence_session.json").read_text(encoding="utf-8"))
    assert session["repair_count"] == 1
    assert len(session["attempts"]) == 2
    assert "resume" in commands[1]
    assert "errors_to_fix" in prompts[1]


def test_s3_failure_feedback_can_early_stop_from_iteration_history(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    history_path = project_root / "experiment" / "results" / "c2c_iteration_history.json"
    write_json(
        history_path,
        {
            "schema_version": "c2c_iteration_history_v1",
            "consecutive_not_viable": 3,
            "best_delta_so_far": -1.08,
            "iterations": [
                {"iteration": 1, "accepted": False},
                {"iteration": 2, "accepted": False},
                {"iteration": 3, "accepted": False},
            ],
        },
    )
    registry = {
        "iteration": 3,
        "max_iterations": 5,
        "status": "running",
        "blocked_reason": None,
        "current_stage": "S3_experiment",
        "stages": {"S3_experiment": {"status": "running", "blocked_reason": None}},
    }
    config = {
        "orchestration": {
            "failure_feedback": {
                "early_stop": {
                    "enabled": True,
                    "min_iterations": 3,
                    "patience": 3,
                    "stop_if_best_delta_below": -0.5,
                }
            }
        }
    }

    routed = Orchestrator._route_s3_failure_to_s1(
        project_root,
        registry,
        {"status": "not_viable"},
        "candidate stayed below baseline",
        config=config,
    )

    assert routed["status"] == "blocked"
    assert "Early-stopped S3 failure feedback" in routed["reason"]
    assert registry["status"] == "blocked"


def test_s3_repairable_proxy_routes_back_to_s2_same_iteration(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    write_json(
        project_root / "experiment" / "results" / "main_results.json",
        {
            "candidate_results": [
                {
                    "id": "idea_a",
                    "decision": "proxy_repairable",
                    "proxy_screen": {"status": "repairable_proxy_risk"},
                    "failure_attribution": {"primary_failure": "repairable_proxy_risk_before_full_training"},
                }
            ]
        },
    )
    registry = {
        "iteration": 2,
        "max_iterations": 5,
        "status": "running",
        "blocked_reason": None,
        "current_stage": "S3_experiment",
        "repair_routes": {},
        "stages": {
            "S1_literature": {"status": "completed", "judge_retries": 0},
            "S2_plan": {"status": "completed", "judge_retries": 1},
            "S3_experiment": {"status": "running", "judge_retries": 1},
            "S4_writing": {"status": "pending", "judge_retries": 0},
            "S5_review": {"status": "pending", "judge_retries": 0},
        },
    }
    config = {"orchestration": {"failure_feedback": {"enabled": True, "route_repairable_proxy_to_s2": True}}}

    assert Orchestrator._should_route_s3_repairable_proxy_to_s2(config, project_root, registry) is True
    routed = Orchestrator._route_s3_repairable_proxy_to_s2(project_root, registry, {"status": "not_viable"}, "proxy repair")

    assert routed["status"] == "routed"
    assert routed["next_stage"] == "S2_plan"
    assert routed["iteration"] == 2
    assert registry["current_stage"] == "S2_plan"
    assert registry["repair_routes"]["2"] == 1


def test_s3_repairable_proxy_allows_three_routes_per_iteration(tmp_path: Path) -> None:
    project_root = tmp_path / "proj_proxy_route_budget"
    write_json(
        project_root / "experiment" / "results" / "main_results.json",
        {
            "candidate_results": [
                {
                    "decision": "proxy_repairable",
                    "proxy_screen": {"status": "repairable_proxy_risk", "soft_fail": True},
                    "failure_attribution": {"primary_failure": "repairable_proxy_risk_before_full_training"},
                }
            ]
        },
    )
    config = {
        "orchestration": {
            "failure_feedback": {
                "enabled": True,
                "route_repairable_proxy_to_s2": True,
                "max_proxy_repair_routes_per_iteration": 3,
            }
        }
    }
    registry = {"iteration": 1, "repair_routes": {"1": 2}}

    assert Orchestrator._should_route_s3_repairable_proxy_to_s2(config, project_root, registry) is True
    registry["repair_routes"]["1"] = 3
    assert Orchestrator._should_route_s3_repairable_proxy_to_s2(config, project_root, registry) is False


def test_s3_blocked_repairable_proxy_routes_before_final_block(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "proj_blocked_route"
    config = _test_config(tmp_path, simulate=False)
    config["orchestration"]["failure_feedback"] = {"enabled": True, "route_repairable_proxy_to_s2": True}
    config["orchestration"]["judge_max_retries"] = 0
    config["review"]["max_iterations"] = 1
    project_root.mkdir(parents=True)
    (project_root / "meta").mkdir()
    (project_root / "literature").mkdir()
    (project_root / "experiment" / "results").mkdir(parents=True)
    write_json(project_root / "meta" / "project_config.yaml", config)
    registry = {
        "project_id": "proj_blocked_route",
        "research_topic": "topic",
        "current_stage": "S3_experiment",
        "iteration": 1,
        "max_iterations": 1,
        "status": "running",
        "blocked_reason": None,
        "stages": {
            "S1_literature": {"status": "completed", "started_at": None, "completed_at": None, "judge_passed": True, "judge_retries": 0, "artifacts": [], "blocked_reason": None},
            "S2_plan": {"status": "completed", "started_at": None, "completed_at": None, "judge_passed": True, "judge_retries": 0, "artifacts": [], "blocked_reason": None},
            "S3_experiment": {"status": "pending", "started_at": None, "completed_at": None, "judge_passed": False, "judge_retries": 0, "artifacts": [], "blocked_reason": None},
            "S4_writing": {"status": "pending", "started_at": None, "completed_at": None, "judge_passed": False, "judge_retries": 0, "artifacts": [], "blocked_reason": None},
            "S5_review": {"status": "pending", "started_at": None, "completed_at": None, "judge_passed": False, "judge_retries": 0, "artifacts": [], "blocked_reason": None},
        },
    }
    write_json(project_root / "meta" / "registry.yaml", registry)
    write_json(
        project_root / "experiment" / "results" / "main_results.json",
        {
            "candidate_results": [
                {
                    "id": "idea_a",
                    "decision": "proxy_repairable",
                    "proxy_screen": {"status": "repairable_proxy_risk"},
                    "failure_attribution": {"primary_failure": "repairable_proxy_risk_before_full_training"},
                }
            ]
        },
    )
    write_json(project_root / "literature" / "ideas.json", [{"title": "idea"}])

    def fake_run(self):
        return {
            "status": "blocked",
            "blocked_reason": "C2C cheap proxy found repairable S2.5 patch risk for all candidates; reroute to S2.5 patch repair before full S3.",
            "artifacts": ["experiment/results/main_results.json"],
        }

    monkeypatch.setattr(config_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module.ExperimentAgent, "run", fake_run)

    result = Orchestrator().start(project_root.name)

    saved = yaml.safe_load((project_root / "meta" / "registry.yaml").read_text(encoding="utf-8"))
    assert result["status"] == "blocked"
    assert result["stage"] == "S3_experiment"
    assert saved["current_stage"] == "S3_experiment"
    assert saved["status"] == "blocked"
    assert saved["repair_routes"]["1"] == 1
    session_log = (project_root / "meta" / "session_log.jsonl").read_text(encoding="utf-8")
    assert "s3_repairable_proxy_to_s2" in session_log


def test_s3_blocked_proxy_rejected_routes_to_s2_same_direction(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "proj_proxy_rejected_route"
    config = _test_config(tmp_path, simulate=False)
    config["orchestration"]["failure_feedback"] = {
        "enabled": True,
        "route_s3_failure_to_s1": True,
        "route_repairable_proxy_to_s2": True,
        "route_proxy_rejected_to_s2": True,
        "max_same_direction_proxy_failures": 5,
    }
    config["orchestration"]["judge_max_retries"] = 0
    config["review"]["max_iterations"] = 2
    project_root.mkdir(parents=True)
    (project_root / "meta").mkdir()
    (project_root / "literature").mkdir()
    (project_root / "experiment" / "results").mkdir(parents=True)
    write_json(project_root / "meta" / "project_config.yaml", config)
    registry = {
        "project_id": "proj_proxy_rejected_route",
        "research_topic": "topic",
        "current_stage": "S3_experiment",
        "iteration": 1,
        "max_iterations": 2,
        "status": "running",
        "blocked_reason": None,
        "stages": {
            "S1_literature": {"status": "completed", "started_at": None, "completed_at": None, "judge_passed": True, "judge_retries": 0, "artifacts": [], "blocked_reason": None},
            "S2_plan": {"status": "completed", "started_at": None, "completed_at": None, "judge_passed": True, "judge_retries": 0, "artifacts": [], "blocked_reason": None},
            "S3_experiment": {"status": "pending", "started_at": None, "completed_at": None, "judge_passed": False, "judge_retries": 0, "artifacts": [], "blocked_reason": None},
            "S4_writing": {"status": "pending", "started_at": None, "completed_at": None, "judge_passed": False, "judge_retries": 0, "artifacts": [], "blocked_reason": None},
            "S5_review": {"status": "pending", "started_at": None, "completed_at": None, "judge_passed": False, "judge_retries": 0, "artifacts": [], "blocked_reason": None},
        },
    }
    write_json(project_root / "meta" / "registry.yaml", registry)
    write_json(
        project_root / "experiment" / "results" / "main_results.json",
        {
            "acceptance": {"passed": False, "reason": "proxy rejected"},
            "candidate_results": [
                {
                    "id": "idea_a",
                    "title": "same direction candidate",
                    "decision": "proxy_rejected",
                    "command_status": "proxy_rejected",
                    "patch_result": {
                        "status": "ok",
                        "changed_files": ["rosetta/model/projector.py"],
                        "validation": {
                            "checks": [
                                {"name": "runtime_smoke:first_batch_train", "returncode": 0},
                                {"name": "py_compile", "returncode": 0},
                            ]
                        },
                    },
                    "proxy_screen": {
                        "status": "rejected",
                        "reason": "proxy mean delta -1.2 below hard threshold -0.3",
                        "proxy_delta_vs_baseline": -1.2,
                        "proxy_dataset_deltas": {"mmlu-redux": -1.8, "ai2-arc": -0.4, "openbookqa": 0.2},
                        "proxy_dataset_regressions": {"mmlu-redux": 1.8, "ai2-arc": 0.4},
                        "proxy_worst_dataset_regression": 1.8,
                        "proxy_score": -2.1,
                        "patch_risk": {
                            "risk_files": ["rosetta/model/projector.py"],
                            "risk_labels": ["projector_mechanism_changed"],
                        },
                    },
                    "failure_attribution": {
                        "primary_failure": "cheap_proxy_rejected_before_full_training",
                        "dragging_datasets": [{"dataset": "mmlu-redux", "delta": -1.8, "regression": 1.8}],
                        "patch_risk": {
                            "risk_files": ["rosetta/model/projector.py"],
                            "risk_labels": ["projector_mechanism_changed"],
                        },
                    },
                }
            ]
        },
    )
    write_json(project_root / "literature" / "ideas.json", [{"title": "idea"}])

    def fake_run(self):
        return {
            "status": "blocked",
            "blocked_reason": "C2C cheap proxy rejected all candidates before full S3; inspect proxy_screen and failure_attribution fields.",
            "artifacts": ["experiment/results/main_results.json"],
        }

    def stop_after_route(self):
        raise RuntimeError("stop after confirming reroute")

    monkeypatch.setattr(config_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module.ExperimentAgent, "run", fake_run)
    monkeypatch.setattr(orchestrator_module.PlanAgent, "run", stop_after_route)

    try:
        Orchestrator().start(project_root.name)
    except RuntimeError as exc:
        assert "stop after confirming reroute" in str(exc)

    saved = yaml.safe_load((project_root / "meta" / "registry.yaml").read_text(encoding="utf-8"))
    assert saved["current_stage"] == "S2_plan"
    assert saved["status"] == "running"
    assert saved["iteration"] == 1
    assert saved["stages"]["S1_literature"]["status"] == "completed"
    assert saved["stages"]["S2_plan"]["status"] == "running"
    assert saved["proxy_rejected_routes"]["1"] == 1
    feedback = json.loads((project_root / "plan" / "performance_feedback.json").read_text(encoding="utf-8"))
    assert feedback["summary"]["next_action"] == "repair_or_variant_same_direction"
    assert feedback["summary"]["recommended_s2_action"] == "mechanism_repair"
    assert "mixed_dataset_signal" in feedback["summary"]["repair_vs_variant_signals"]
    assert feedback["candidate_results"][0]["dragging_datasets"][0]["dataset"] == "mmlu-redux"
    assert feedback["candidate_results"][0]["runtime_validation"]["runtime_smoke"] == "passed"
    session_log = (project_root / "meta" / "session_log.jsonl").read_text(encoding="utf-8")
    assert "s3_proxy_rejected_to_s2" in session_log


def test_s3_proxy_feedback_recommends_patch_repair_for_runtime_failure(tmp_path: Path) -> None:
    project_root = tmp_path / "proj_proxy_runtime_feedback"
    write_json(
        project_root / "experiment" / "results" / "main_results.json",
        {
            "candidate_results": [
                {
                    "id": "idea_a",
                    "decision": "proxy_rejected",
                    "command_status": "proxy_rejected",
                    "patch_result": {
                        "status": "ok",
                        "changed_files": ["rosetta/model/projector.py"],
                        "validation": {
                            "checks": [
                                {"name": "runtime_smoke:first_batch_train", "returncode": 1, "failure_category": "dtype_mismatch"},
                            ]
                        },
                    },
                    "proxy_screen": {
                        "status": "rejected",
                        "proxy_delta_vs_baseline": -0.5,
                        "proxy_dataset_deltas": {"mmlu-redux": 0.1, "ai2-arc": -0.7, "openbookqa": -0.9},
                        "patch_risk": {"risk_files": ["rosetta/model/projector.py"], "risk_labels": ["projector_mechanism_changed"]},
                    },
                    "failure_attribution": {"primary_failure": "cheap_proxy_rejected_before_full_training"},
                }
            ]
        },
    )
    registry = {"iteration": 1}

    feedback = orchestrator_module._s3_proxy_performance_feedback(
        project_root,
        registry,
        {"status": "blocked"},
        "proxy rejected",
        route_count=1,
        failure_count=1,
        max_failures=5,
    )

    assert feedback["summary"]["recommended_s2_action"] == "patch_repair"
    assert "runtime_or_validation_failed" in feedback["summary"]["repair_vs_variant_signals"]


def test_s3_proxy_feedback_recommends_variant_for_all_dataset_collapse(tmp_path: Path) -> None:
    project_root = tmp_path / "proj_proxy_variant_feedback"
    write_json(
        project_root / "experiment" / "results" / "main_results.json",
        {
            "candidate_results": [
                {
                    "id": "idea_a",
                    "decision": "proxy_rejected",
                    "command_status": "proxy_rejected",
                    "patch_result": {
                        "status": "ok",
                        "changed_files": ["rosetta/model/projector.py"],
                        "validation": {"checks": [{"name": "runtime_smoke:first_batch_train", "returncode": 0}]},
                    },
                    "proxy_screen": {
                        "status": "rejected",
                        "proxy_delta_vs_baseline": -3.2,
                        "proxy_dataset_deltas": {"mmlu-redux": -3.0, "ai2-arc": -2.8, "openbookqa": -3.6},
                        "patch_risk": {"risk_files": ["rosetta/model/projector.py"], "risk_labels": ["projector_mechanism_changed"]},
                    },
                    "failure_attribution": {"primary_failure": "cheap_proxy_rejected_before_full_training"},
                }
            ]
        },
    )
    registry = {"iteration": 1}

    feedback = orchestrator_module._s3_proxy_performance_feedback(
        project_root,
        registry,
        {"status": "blocked"},
        "proxy rejected",
        route_count=1,
        failure_count=1,
        max_failures=5,
    )

    assert feedback["summary"]["recommended_s2_action"] == "new_same_direction_variant"
    assert "all_proxy_datasets_below_baseline" in feedback["summary"]["repair_vs_variant_signals"]


def test_s3_direction_scorecard_accumulates_direction_evidence(tmp_path: Path) -> None:
    project_root = tmp_path / "proj_direction_scorecard"
    (project_root / "plan").mkdir(parents=True)
    (project_root / "literature").mkdir(parents=True)
    write_json(
        project_root / "literature" / "ideas.json",
        [
            {
                "id": "utility_predicted_cache_routing",
                "title": "Utility-predicted cache routing",
                "mechanism_type": "utility_predicted_cache_routing",
                "selected": True,
            }
        ],
    )
    write_json(
        project_root / "plan" / "candidate_ideas.json",
        [
            {
                "id": "utility_variant_a",
                "title": "Utility variant A",
                "mechanism_type": "utility_predicted_cache_routing",
                "selected": True,
                "s2_planner": {"s1_direction_id": "utility_predicted_cache_routing"},
            }
        ],
    )
    write_json(
        project_root / "experiment" / "results" / "main_results.json",
        {
            "candidate_results": [
                {
                    "id": "utility_variant_a",
                    "decision": "proxy_rejected",
                    "patch_result": {
                        "status": "ok",
                        "changed_files": ["rosetta/model/projector.py"],
                        "validation": {"checks": [{"name": "runtime_smoke:first_batch_train", "returncode": 0}]},
                    },
                    "proxy_screen": {
                        "status": "rejected",
                        "proxy_delta_vs_baseline": -0.4,
                        "proxy_dataset_deltas": {"mmlu-redux": 0.2, "ai2-arc": -0.7, "openbookqa": -0.4},
                        "patch_risk": {"risk_files": ["rosetta/model/projector.py"], "risk_labels": ["projector_mechanism_changed"]},
                    },
                    "failure_attribution": {"dragging_datasets": [{"dataset": "ai2-arc", "regression": 0.7}]},
                }
            ]
        },
    )
    registry = {"iteration": 1}
    feedback = orchestrator_module._s3_proxy_performance_feedback(
        project_root,
        registry,
        {"status": "blocked"},
        "proxy rejected",
        route_count=1,
        failure_count=1,
        max_failures=5,
    )

    scorecard = orchestrator_module._update_c2c_direction_scorecard(project_root, registry, feedback)

    current = scorecard["current_direction"]
    assert current["direction_id"] == "utility_predicted_cache_routing"
    assert current["summary"]["best_proxy_delta"] == -0.4
    assert current["summary"]["positive_dataset_signal_attempts"] == 1
    assert current["summary"]["runtime_stable_attempts"] == 1
    assert current["summary"]["low_patch_risk_attempts"] == 1
    assert current["summary"]["all_dataset_collapse_attempts"] == 0
    assert current["s1_feedback"]["recommendation"] == "keep_direction_with_targeted_repair"
    saved = json.loads((project_root / "plan" / "direction_scorecard.json").read_text(encoding="utf-8"))
    assert saved["current_direction_id"] == "utility_predicted_cache_routing"


def test_s3_proxy_rejected_routes_to_s1_after_same_direction_budget(tmp_path: Path) -> None:
    project_root = tmp_path / "proj_proxy_rejected_budget"
    write_json(
        project_root / "experiment" / "results" / "main_results.json",
        {
            "candidate_results": [
                {
                    "id": "idea_a",
                    "decision": "proxy_rejected",
                    "proxy_screen": {"status": "rejected", "proxy_delta_vs_baseline": -2.0},
                    "failure_attribution": {"primary_failure": "cheap_proxy_rejected_before_full_training"},
                }
            ]
        },
    )
    config = {
        "orchestration": {
            "failure_feedback": {
                "enabled": True,
                "route_s3_failure_to_s1": True,
                "route_proxy_rejected_to_s2": True,
                "max_same_direction_proxy_failures": 5,
            }
        }
    }
    registry = {
        "iteration": 1,
        "max_iterations": 3,
        "status": "running",
        "blocked_reason": None,
        "current_stage": "S3_experiment",
        "proxy_rejected_routes": {"1": 4},
        "stages": {
            "S1_literature": {"status": "completed", "judge_retries": 0},
            "S2_plan": {"status": "completed", "judge_retries": 0},
            "S3_experiment": {"status": "running", "judge_retries": 0},
            "S4_writing": {"status": "pending", "judge_retries": 0},
            "S5_review": {"status": "pending", "judge_retries": 0},
        },
    }

    assert Orchestrator._should_route_s3_proxy_rejected_to_s2(config, project_root, registry) is False
    assert Orchestrator._should_route_s3_proxy_rejected_to_s1(config, project_root, registry) is True
    routed = Orchestrator._route_s3_failure_to_s1(project_root, registry, {"status": "blocked"}, "proxy rejected budget exhausted", config=config)

    assert routed["status"] == "routed"
    assert registry["iteration"] == 2
    assert registry["current_stage"] == "S1_literature"
    assert "s3_failure_feedback" in registry["invalidated_by"]


def test_s3_proxy_rejected_legacy_route_to_s1_when_same_direction_disabled(tmp_path: Path) -> None:
    project_root = tmp_path / "proj_proxy_rejected_legacy"
    write_json(
        project_root / "experiment" / "results" / "main_results.json",
        {"candidate_results": [{"id": "idea_a", "decision": "proxy_rejected"}]},
    )
    config = {
        "orchestration": {
            "failure_feedback": {
                "enabled": True,
                "route_s3_failure_to_s1": True,
                "route_proxy_rejected_to_s2": False,
            }
        }
    }
    registry = {"iteration": 1, "proxy_rejected_routes": {}}

    assert Orchestrator._should_route_s3_proxy_rejected_to_s2(config, project_root, registry) is False
    assert Orchestrator._should_route_s3_proxy_rejected_to_s1(config, project_root, registry) is True

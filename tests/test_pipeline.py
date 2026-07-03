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
from auto_research.registry import block_stage
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


def _write_direction_and_variant_contracts(project_root: Path) -> list[str]:
    direction = {
        "schema_version": "auto_research_direction_v1",
        "direction_id": "utility_predicted_cache_routing",
        "title": "Utility-predicted cache routing",
        "mechanism_type": "utility_predicted_cache_routing",
        "mechanism_axis": "routing",
        "integration_point": "projector",
        "control_signal": "utility",
        "hypothesis": "Predict transferred-cache utility before routing.",
        "why_baseline_fails": "The baseline lacks downstream utility control.",
        "expected_metric_signature": {"primary_metric": "three_dataset_mean", "expected_direction": "increase"},
        "required_evidence_refs": [{"source_type": "code", "source_label": "rosetta/model/projector.py", "claim": "surface"}],
        "counterevidence_refs": [{"source_type": "failure_feedback", "source_label": "risk", "claim": "avoid hard gates"}],
        "implementation_surface_refs": [{"source_type": "code", "source_label": "rosetta/model/projector.py", "claim": "surface"}],
        "known_negative_memory_refs": [],
        "go_to_s2_conditions": ["evidence resolved"],
        "return_to_s1_conditions": ["budget exhausted"],
        "expected_files": ["rosetta/model/projector.py"],
        "verification_commands": ["py_compile"],
        "used_shared_memory_refs": [],
    }
    variant = {
        "id": "utility_predicted_cache_routing",
        "title": "Utility-predicted cache routing",
        "variant_fingerprint": "fp_utility_router",
        "mechanism_axis": "routing",
        "integration_point": "projector",
        "control_signal": "utility",
    }
    write_json(project_root / "literature" / "direction.json", direction)
    write_json(
        project_root / "plan" / "planner_decision.json",
        {
            "schema_version": "auto_research_planner_decision_v1",
            "direction_id": direction["direction_id"],
            "planner_summary": "Mocked S2 planner decision.",
            "planning_mode": "same_direction_variant",
            "used_shared_memory_refs": [],
            "next_variant": variant,
        },
    )
    write_json(
        project_root / "plan" / "variant_contract.json",
        {
            "schema_version": "auto_research_variant_contract_v1",
            "direction_id": direction["direction_id"],
            "variant_id": variant["id"],
            "title": variant["title"],
            "mode": "regular",
            "variant_fingerprint": variant["variant_fingerprint"],
            "mechanism_axis": "routing",
            "integration_point": "projector",
            "control_signal": "utility",
            "hypothesis": direction["hypothesis"],
            "why_next": "Mocked retryable S2 plan.",
            "expected_files": ["rosetta/model/projector.py"],
            "implementation_surface_refs": direction["implementation_surface_refs"],
            "resource_budget": {},
            "expected_metric_signature": direction["expected_metric_signature"],
            "ablation": {"switch": "disable_utility_router", "control": "ablation-off"},
            "acceptance": {"min_delta_to_pass": 0.1, "max_dataset_regression": 2.0},
            "failure_routing": {
                "go_to_s3_conditions": ["gate passes"],
                "return_to_s2_conditions": ["patch invalid"],
                "return_to_s1_conditions": ["budget exhausted"],
            },
            "used_shared_memory_refs": [],
        },
    )
    write_json(
        project_root / "plan" / "variant_fingerprint.json",
        {
            "schema_version": "auto_research_variant_fingerprint_v1",
            "direction_id": direction["direction_id"],
            "variant_id": variant["id"],
            "variant_fingerprint": variant["variant_fingerprint"],
            "mechanism_axis": "routing",
            "integration_point": "projector",
            "control_signal": "utility",
            "history_fingerprints": [],
            "is_repeat": False,
            "mode": "regular",
        },
    )
    return ["plan/planner_decision.json", "plan/variant_contract.json", "plan/variant_fingerprint.json"]


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
    assert any(item["path"] == "literature/direction.json" and item["exists"] for item in s1_contract["produced_outputs"])
    assert any(item["path"] == "literature/ideas.json" and item["exists"] for item in s1_contract["produced_outputs"])
    assert (tmp_path / project_id / "literature" / "direction.json").exists()
    assert (tmp_path / project_id / "literature" / "direction_scorecard.json").exists()
    assert (tmp_path / project_id / "literature" / "novelty_audit.json").exists()
    assert (tmp_path / project_id / "literature" / "evidence_session.json").exists()
    assert (tmp_path / project_id / "plan" / "planner_decision.json").exists()
    assert (tmp_path / project_id / "plan" / "variant_contract.json").exists()
    assert (tmp_path / project_id / "plan" / "variant_fingerprint.json").exists()
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


def test_s3_full_failure_records_proxy_calibration_shared_memory(tmp_path: Path) -> None:
    project_root = tmp_path / "proj_full_proxy_memory"
    write_json(
        project_root / "experiment" / "results" / "main_results.json",
        {
            "baseline": {"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
            "acceptance": {"passed": False, "reason": "below baseline"},
            "candidate_results": [
                {
                    "id": "proxy_pass_full_fail",
                    "title": "Proxy pass full fail",
                    "mechanism_type": "utility_predicted_cache_routing",
                    "decision": "not_viable",
                    "metrics": {"mean": 49.8, "datasets": {"mmlu-redux": 48.5}},
                    "delta_vs_baseline": -0.2,
                    "proxy_screen": {
                        "status": "passed",
                        "proxy_delta_vs_baseline": 0.8,
                        "proxy_dataset_deltas": {"mmlu-redux": 0.9},
                    },
                }
            ],
        },
    )
    write_json(
        project_root / "experiment" / "results" / "proxy_calibration.json",
        {
            "summary": {
                "candidate_count": 1,
                "proxy_false_positive_count": 1,
                "proxy_false_positive_rate": 1.0,
                "dataset_error_summary": {"mmlu-redux": {"misprediction_count": 1, "count": 1}},
                "mechanism_false_positive_summary": {
                    "utility_predicted_cache_routing": {"count": 1, "false_positive_count": 1, "false_positive_rate": 1.0}
                },
            },
            "current_iteration": {
                "iteration": 1,
                "acceptance_passed": False,
                "candidate_count": 1,
                "proxy_false_positive_count": 1,
                "proxy_false_positive_rate": 1.0,
                "candidates": [
                    {
                        "id": "proxy_pass_full_fail",
                        "mechanism_type": "utility_predicted_cache_routing",
                        "proxy_false_positive": True,
                        "mispredicted_datasets": ["mmlu-redux"],
                    }
                ],
            },
        },
    )
    config = {
        "orchestration": {
            "failure_feedback": {"enabled": True, "route_s3_failure_to_s1": True},
            "shared_method_memory": {
                "enabled": True,
                "path": str(tmp_path / "method_memory.jsonl"),
                "summary_path": str(tmp_path / "method_memory.md"),
            },
        }
    }
    registry = {
        "iteration": 1,
        "max_iterations": 3,
        "status": "running",
        "blocked_reason": None,
        "current_stage": "S3_experiment",
        "stages": {
            "S1_literature": {"status": "completed", "judge_retries": 0},
            "S2_plan": {"status": "completed", "judge_retries": 0},
            "S3_experiment": {"status": "running", "judge_retries": 0},
            "S4_writing": {"status": "pending", "judge_retries": 0},
            "S5_review": {"status": "pending", "judge_retries": 0},
        },
    }

    routed = Orchestrator._route_s3_failure_to_s1(project_root, registry, {"status": "not_viable"}, "below baseline", config=config)
    memory = json.loads((tmp_path / "method_memory.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert routed["status"] == "routed"
    assert routed["shared_method_memory"]["status"] == "appended"
    assert memory["proxy_calibration"]["overall_false_positive_count"] == 1
    assert "proxy_full_false_positive" in memory["memory_quality"]["signals"]
    assert memory["memory_quality"]["priority"] > 5


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


def test_s2_retryable_codex_limit_pauses_without_consuming_judge_retry(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "proj_retryable_pause"
    config = _test_config(tmp_path, simulate=False)
    config["orchestration"]["judge_max_retries"] = 0
    project_root.mkdir(parents=True)
    (project_root / "meta").mkdir()
    (project_root / "plan" / "code_patches").mkdir(parents=True)
    write_json(project_root / "meta" / "project_config.yaml", config)
    registry = {
        "project_id": "proj_retryable_pause",
        "research_topic": "topic",
        "current_stage": "S2_plan",
        "iteration": 1,
        "max_iterations": 2,
        "status": "running",
        "blocked_reason": None,
        "stages": {
            "S0_intake": {"status": "completed", "started_at": None, "completed_at": None, "judge_passed": True, "judge_retries": 0, "artifacts": [], "blocked_reason": None},
            "S1_literature": {"status": "completed", "started_at": None, "completed_at": None, "judge_passed": True, "judge_retries": 0, "artifacts": [], "blocked_reason": None},
            "S2_plan": {"status": "pending", "started_at": None, "completed_at": None, "judge_passed": False, "judge_retries": 0, "artifacts": [], "blocked_reason": None},
            "S3_experiment": {"status": "pending", "started_at": None, "completed_at": None, "judge_passed": False, "judge_retries": 0, "artifacts": [], "blocked_reason": None},
            "S4_writing": {"status": "pending", "started_at": None, "completed_at": None, "judge_passed": False, "judge_retries": 0, "artifacts": [], "blocked_reason": None},
            "S5_review": {"status": "pending", "started_at": None, "completed_at": None, "judge_passed": False, "judge_retries": 0, "artifacts": [], "blocked_reason": None},
        },
    }
    write_json(project_root / "meta" / "registry.yaml", registry)
    write_json(
        project_root / "plan" / "code_patches" / "patch_manifest.json",
        {
            "status": "retryable_no_valid_patch",
            "retryable": True,
            "retryable_patch_count": 1,
            "valid_patch_count": 0,
            "candidates": [
                {
                    "candidate_id": "rate_limited",
                    "status": "retryable_codex_failed",
                    "retryable": True,
                    "failure_category": "llm_rate_limit_or_quota",
                    "reason": "429 Too Many Requests",
                }
            ],
        },
    )

    def fake_plan_run(self):
        plan_dir = self.context.project_root / "plan"
        write_yaml(
            plan_dir / "plan.yaml",
            {
                "hypotheses": [{"id": "h1"}],
                "baselines": [{"name": "base"}, {"name": "candidate"}],
                "datasets": [{"name": "mmlu-redux"}],
                "task_graph": {},
                "resource_budget": {},
                "execution": {
                    "collector": "c2c_small_loop",
                    "min_delta_to_pass": 0.1,
                    "max_dataset_regression": 2.0,
                },
                "acceptance_criteria": {
                    "minimum_mean_delta": 0.1,
                    "coverage_diagnostics_required": True,
                    "matched_coverage_ablation_required": True,
                },
                "ablation_matrix": [{"experiment": "matched transfer coverage control", "matched_coverage_ablation": {"required": True}}],
                "reviewer_risk_controls": {"top_concerns": []},
            },
        )
        write_yaml(plan_dir / "short_loop_plan.yaml", {"run": True})
        write_json(
            plan_dir / "candidate_ideas.json",
            [
                {
                    "id": "utility_predicted_cache_routing",
                    "title": "Utility-predicted cache routing",
                    "selected": True,
                    "hypothesis": "Predict transferred-cache utility before routing.",
                    "description": "Mechanism-level utility prediction for cache routing.",
                    "mechanism_type": "utility_predicted_cache_routing",
                    "mechanism_contract": {"components": ["utility predictor"], "ablation_switch": "disable_utility_router"},
                    "experiment_contract": {"ablation_switch": "disable_utility_router"},
                    "implementation_scope": {"files": ["rosetta/model/projector.py"]},
                    "decomposition_plan": [{"step": "wire utility router"}],
                    "expected_signature": {"stats": ["accepted_span_rate"]},
                }
            ],
        )
        variant_artifacts = _write_direction_and_variant_contracts(self.context.project_root)
        return {"artifacts": ["plan/plan.yaml", *variant_artifacts, "plan/short_loop_plan.yaml", "plan/candidate_ideas.json", "plan/code_patches/patch_manifest.json"]}

    monkeypatch.setattr(config_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module.PlanAgent, "run", fake_plan_run)

    result = Orchestrator().start(project_root.name)

    saved = yaml.safe_load((project_root / "meta" / "registry.yaml").read_text(encoding="utf-8"))
    assert result["status"] == "retryable_paused"
    assert result["stage"] == "S2_plan"
    assert result["pause_type"] == "codex_quota_or_rate_limit"
    assert "resume --project-id proj_retryable_pause" in result["resume_instruction"]
    assert saved["status"] == "retryable_paused"
    assert saved["current_stage"] == "S2_plan"
    assert saved["stages"]["S2_plan"]["status"] == "retryable_paused"
    assert saved["stages"]["S2_plan"]["judge_retries"] == 0
    state = json.loads((project_root / "orchestration" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "retryable_paused"
    assert state["current_stage"] == "S2_plan"
    assert state["resume_instruction"] == saved["resume_instruction"]
    contract = json.loads((project_root / "orchestration" / "stage_contracts" / "S2_plan.json").read_text(encoding="utf-8"))
    assert contract["status"] == "retryable_paused"
    assert contract["gate"]["status"] == "NEEDS_RETRY"
    session_log = (project_root / "meta" / "session_log.jsonl").read_text(encoding="utf-8")
    assert "retryable_paused" in session_log


def test_s2_runtime_smoke_resource_retry_pauses_without_codex_repair(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "proj_resource_retry_pause"
    config = _test_config(tmp_path, simulate=False)
    config["orchestration"]["judge_max_retries"] = 0
    project_root.mkdir(parents=True)
    (project_root / "meta").mkdir()
    (project_root / "plan" / "code_patches").mkdir(parents=True)
    write_json(project_root / "meta" / "project_config.yaml", config)
    registry = {
        "project_id": "proj_resource_retry_pause",
        "research_topic": "topic",
        "current_stage": "S2_plan",
        "iteration": 1,
        "max_iterations": 2,
        "status": "running",
        "blocked_reason": None,
        "stages": {
            "S0_intake": {"status": "completed", "started_at": None, "completed_at": None, "judge_passed": True, "judge_retries": 0, "artifacts": [], "blocked_reason": None},
            "S1_literature": {"status": "completed", "started_at": None, "completed_at": None, "judge_passed": True, "judge_retries": 0, "artifacts": [], "blocked_reason": None},
            "S2_plan": {"status": "pending", "started_at": None, "completed_at": None, "judge_passed": False, "judge_retries": 0, "artifacts": [], "blocked_reason": None},
            "S3_experiment": {"status": "pending", "started_at": None, "completed_at": None, "judge_passed": False, "judge_retries": 0, "artifacts": [], "blocked_reason": None},
            "S4_writing": {"status": "pending", "started_at": None, "completed_at": None, "judge_passed": False, "judge_retries": 0, "artifacts": [], "blocked_reason": None},
            "S5_review": {"status": "pending", "started_at": None, "completed_at": None, "judge_passed": False, "judge_retries": 0, "artifacts": [], "blocked_reason": None},
        },
    }
    write_json(project_root / "meta" / "registry.yaml", registry)
    write_json(
        project_root / "plan" / "code_patches" / "patch_manifest.json",
        {
            "status": "retryable_no_valid_patch",
            "retryable": True,
            "retryable_patch_count": 1,
            "valid_patch_count": 0,
            "candidates": [
                {
                    "candidate_id": "resource_wait",
                    "status": "validation_failed",
                    "retryable": True,
                    "resource_retry": True,
                    "failure_category": "runtime_smoke_resource_retry",
                    "reason": "runtime smoke could not obtain a GPU with enough free memory",
                }
            ],
        },
    )

    def fake_plan_run(self):
        plan_dir = self.context.project_root / "plan"
        write_yaml(
            plan_dir / "plan.yaml",
            {
                "hypotheses": [{"id": "h1"}],
                "baselines": [{"name": "base"}, {"name": "candidate"}],
                "datasets": [{"name": "mmlu-redux"}],
                "task_graph": {},
                "resource_budget": {},
                "execution": {
                    "collector": "c2c_small_loop",
                    "min_delta_to_pass": 0.1,
                    "max_dataset_regression": 2.0,
                },
                "acceptance_criteria": {
                    "minimum_mean_delta": 0.1,
                    "coverage_diagnostics_required": True,
                    "matched_coverage_ablation_required": True,
                },
                "ablation_matrix": [{"experiment": "matched transfer coverage control", "matched_coverage_ablation": {"required": True}}],
                "reviewer_risk_controls": {"top_concerns": []},
            },
        )
        write_yaml(plan_dir / "short_loop_plan.yaml", {"run": True})
        write_json(
            plan_dir / "candidate_ideas.json",
            [
                {
                    "id": "utility_predicted_cache_routing",
                    "title": "Utility-predicted cache routing",
                    "selected": True,
                    "hypothesis": "Predict transferred-cache utility before routing.",
                    "description": "Mechanism-level utility prediction for cache routing.",
                    "mechanism_type": "utility_predicted_cache_routing",
                    "mechanism_contract": {"components": ["utility predictor"], "ablation_switch": "disable_utility_router"},
                    "experiment_contract": {"ablation_switch": "disable_utility_router"},
                    "implementation_scope": {"files": ["rosetta/model/projector.py"]},
                    "decomposition_plan": [{"step": "wire utility router"}],
                    "expected_signature": {"stats": ["accepted_span_rate"]},
                }
            ],
        )
        variant_artifacts = _write_direction_and_variant_contracts(self.context.project_root)
        return {"artifacts": ["plan/plan.yaml", *variant_artifacts, "plan/short_loop_plan.yaml", "plan/candidate_ideas.json", "plan/code_patches/patch_manifest.json"]}

    monkeypatch.setattr(config_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module.PlanAgent, "run", fake_plan_run)

    result = Orchestrator().start(project_root.name)

    saved = yaml.safe_load((project_root / "meta" / "registry.yaml").read_text(encoding="utf-8"))
    assert result["status"] == "retryable_paused"
    assert result["pause_type"] == "runtime_smoke_resource_retry"
    assert "GPU memory" in result["reason"]
    assert saved["status"] == "retryable_paused"
    assert saved["pause_type"] == "runtime_smoke_resource_retry"
    assert saved["stages"]["S2_plan"]["judge_retries"] == 0


def test_s3_implementation_failure_routes_to_s2_without_consuming_direction_budget(tmp_path: Path) -> None:
    project_root = tmp_path / "proj_implementation_failure_route"
    write_json(
        project_root / "experiment" / "results" / "main_results.json",
        {
            "candidate_results": [
                {
                    "id": "idea_impl",
                    "decision": "proxy_repairable",
                    "command_status": "proxy_repairable",
                    "patch_result": {
                        "status": "applied",
                        "changed_files": ["rosetta/model/projector.py"],
                        "validation": {
                            "checks": [
                                {
                                    "name": "runtime_smoke:mechanism_activation_wiring",
                                    "returncode": 1,
                                    "failure_category": "mechanism_activation_wiring_failed",
                                }
                            ]
                        },
                    },
                    "proxy_screen": {
                        "status": "repairable_proxy_risk",
                        "reason": "ablation switch produced no observable proxy eval metric or prediction change",
                        "activation_smoke": {
                            "status": "failed",
                            "mechanism_trace": {"status": "missing"},
                        },
                    },
                    "failure_attribution": {"primary_failure": "proxy_activation_smoke_no_effect"},
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
        "repair_routes": {"2": 4},
        "implementation_repair_routes": {},
        "stages": {
            "S1_literature": {"status": "completed", "judge_retries": 0},
            "S2_plan": {"status": "completed", "judge_retries": 1},
            "S3_experiment": {"status": "running", "judge_retries": 1},
            "S4_writing": {"status": "pending", "judge_retries": 0},
            "S5_review": {"status": "pending", "judge_retries": 0},
        },
    }
    config = {
        "orchestration": {
            "failure_feedback": {
                "enabled": True,
                "route_repairable_proxy_to_s2": True,
                "route_s3_failure_to_s1": True,
                "max_same_direction_proxy_failures": 5,
            }
        }
    }

    assert orchestrator_module._s3_feedback_failure_class(project_root) == "implementation_failure"
    assert Orchestrator._should_route_s3_repairable_proxy_to_s2(config, project_root, registry) is True
    assert Orchestrator._should_route_s3_repairable_proxy_to_s1(config, project_root, registry) is False
    routed = Orchestrator._route_s3_repairable_proxy_to_s2(project_root, registry, {"status": "blocked"}, "activation wiring failed", config=config)

    assert routed["status"] == "routed"
    assert routed["next_stage"] == "S2_plan"
    assert routed["failure_class"] == "implementation_failure"
    assert routed["repair_lane"] == "s2_5_only_implementation_repair"
    assert routed["skips_s2_planner"] is True
    assert routed["same_direction_failure_count"] == 0
    assert registry["repair_routes"]["2"] == 4
    assert registry["implementation_repair_routes"]["2"] == 1
    assert registry["s2_5_repair_dispatch"]["active"] is True
    assert registry["s2_5_repair_dispatch"]["selected_candidate_id"] == "idea_impl"
    assert registry["iteration"] == 2
    assert registry["current_stage"] == "S2_plan"
    assert not (project_root / "plan" / "direction_scorecard.json").exists()
    feedback = json.loads((project_root / "plan" / "performance_feedback.json").read_text(encoding="utf-8"))
    assert feedback["summary"]["failure_class"] == "implementation_failure"
    assert feedback["summary"]["does_not_consume_same_direction_attempt"] is True
    assert feedback["summary"]["recommended_s2_action"] == "patch_repair"
    assert feedback["summary"]["s2_action_policy"]["matched_rule"] == "implementation_failure"
    dispatch = json.loads((project_root / "plan" / "s2_5_repair_dispatch.json").read_text(encoding="utf-8"))
    assert dispatch["mode"] == "s2_5_only_implementation_repair"
    assert dispatch["selected_candidate_id"] == "idea_impl"
    assert dispatch["same_candidate_required"] is True
    assert dispatch["reuse_persistent_codex_session"] is True
    assert dispatch["do_not_replan_method"] is True
    assert dispatch["changed_files"] == ["rosetta/model/projector.py"]
    assert "mechanism_activation_wiring_failed" in dispatch["implementation_failure_signals"]


def test_s3_proxy_oom_pauses_as_resource_retry_not_s2_5_repair(tmp_path: Path) -> None:
    project_root = tmp_path / "proj_s3_proxy_oom_resource_retry"
    write_json(
        project_root / "experiment" / "results" / "main_results.json",
        {
            "candidate_results": [
                {
                    "id": "proxy_oom",
                    "decision": "proxy_repairable",
                    "command_status": "proxy_repairable",
                    "proxy_screen": {
                        "status": "repairable_proxy_risk",
                        "reason": "proxy command 0 failed: resource_oom",
                        "command_failure": {
                            "category": "resource_oom",
                            "summary": "resource_oom: CUDA out of memory",
                        },
                    },
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
                "route_s3_failure_to_s1": True,
            }
        }
    }
    registry = {
        "iteration": 1,
        "max_iterations": 5,
        "status": "running",
        "blocked_reason": None,
        "current_stage": "S3_experiment",
        "repair_routes": {"1": 0},
        "implementation_repair_routes": {"1": 8},
        "implementation_repair_routes_by_candidate": {"1:proxy_oom:unknown_variant": 8},
        "stages": {
            "S1_literature": {"status": "completed", "judge_retries": 0},
            "S2_plan": {"status": "completed", "judge_retries": 0},
            "S3_experiment": {"status": "running", "judge_retries": 0},
            "S4_writing": {"status": "pending", "judge_retries": 0},
            "S5_review": {"status": "pending", "judge_retries": 0},
        },
    }

    assert orchestrator_module._s3_result_has_resource_retry(project_root) is True
    assert orchestrator_module._s3_feedback_failure_class(project_root) == "resource_retry"
    assert Orchestrator._should_route_s3_implementation_failure_to_s2(config, project_root, registry) is False
    assert Orchestrator._should_route_s3_repairable_proxy_to_s2(config, project_root, registry) is False
    pause = Orchestrator._s3_resource_retry_pause_details(
        project_root,
        {"status": "blocked"},
        "C2C cheap proxy found repairable S2.5 patch risk for all candidates",
    )

    assert pause is not None
    assert pause["pause_type"] == "s3_proxy_resource_retry"
    assert "not an S2.5 implementation repair" in pause["reason"]
    assert registry["implementation_repair_routes"]["1"] == 8
    assert registry["implementation_repair_routes_by_candidate"]["1:proxy_oom:unknown_variant"] == 8


@pytest.mark.parametrize(
    ("probe_payload", "expected_signal"),
    [
        (
            {
                "probe_type": "repo_small_batch_forward_failed_static_trace",
                "fallback_reason": "torch_import_failed",
                "failures": ["torch_import_failed"],
            },
            "torch_import_failed",
        ),
        (
            {
                "probe_type": "repo_small_batch_forward_failed_static_trace",
                "fallback_reason": "projector_import_failed",
                "failures": ["projector_import_failed"],
            },
            "projector_import_failed",
        ),
        (
            {
                "probe_type": "repo_small_batch_forward",
                "failures": ["enabled_disabled_forward_tensors_identical"],
                "mechanism_observed": False,
            },
            "enabled_disabled_forward_tensors_identical",
        ),
    ],
)
def test_s3_activation_forward_probe_failures_are_implementation_only(tmp_path: Path, probe_payload: dict, expected_signal: str) -> None:
    project_root = tmp_path / "proj_forward_probe_impl_route"
    write_json(
        project_root / "experiment" / "results" / "main_results.json",
        {
            "candidate_results": [
                {
                    "id": "forward_probe_impl",
                    "decision": "proxy_repairable",
                    "command_status": "proxy_repairable",
                    "patch_result": {
                        "status": "applied",
                        "changed_files": ["rosetta/model/projector.py"],
                        "validation": {
                            "checks": [
                                {
                                    "name": "runtime_smoke:mechanism_activation_forward_probe",
                                    "returncode": 1,
                                    "failure_category": "mechanism_activation_forward_probe_failed",
                                    "forward_probe_diagnostics": {
                                        "switch_seen_by_forward": False,
                                        "projector_output_identical": True,
                                        "changed_tensors": [],
                                        "identical_tensors": ["projector_output"],
                                    },
                                    "probe": probe_payload,
                                }
                            ]
                        },
                    },
                    "proxy_screen": {
                        "status": "repairable_proxy_risk",
                        "reason": "forward probe did not observe mechanism activation",
                    },
                    "failure_attribution": {"primary_failure": "repairable_proxy_risk_before_full_training"},
                }
            ]
        },
    )
    registry = {
        "iteration": 1,
        "max_iterations": 5,
        "status": "running",
        "blocked_reason": None,
        "current_stage": "S3_experiment",
        "repair_routes": {"1": 4},
        "implementation_repair_routes": {},
        "stages": {
            "S1_literature": {"status": "completed", "judge_retries": 0},
            "S2_plan": {"status": "completed", "judge_retries": 0},
            "S3_experiment": {"status": "running", "judge_retries": 0},
            "S4_writing": {"status": "pending", "judge_retries": 0},
            "S5_review": {"status": "pending", "judge_retries": 0},
        },
    }
    config = {
        "orchestration": {
            "failure_feedback": {
                "enabled": True,
                "route_repairable_proxy_to_s2": True,
                "route_s3_failure_to_s1": True,
                "max_same_direction_proxy_failures": 5,
            }
        }
    }

    assert orchestrator_module._s3_feedback_failure_class(project_root) == "implementation_failure"
    assert Orchestrator._should_route_s3_repairable_proxy_to_s2(config, project_root, registry) is True
    routed = Orchestrator._route_s3_repairable_proxy_to_s2(project_root, registry, {"status": "blocked"}, "forward probe failed", config=config)

    assert routed["failure_class"] == "implementation_failure"
    assert routed["same_direction_failure_count"] == 0
    assert registry["repair_routes"]["1"] == 4
    assert registry["implementation_repair_routes"]["1"] == 1
    assert not (project_root / "plan" / "direction_scorecard.json").exists()
    feedback = json.loads((project_root / "plan" / "performance_feedback.json").read_text(encoding="utf-8"))
    assert feedback["summary"]["does_not_consume_same_direction_attempt"] is True
    assert feedback["summary"]["recommended_s2_action"] == "patch_repair"
    assert expected_signal in feedback["summary"]["repair_vs_variant_signals"]
    assert expected_signal in feedback["candidate_results"][0]["implementation_failure_signals"]
    dispatch = json.loads((project_root / "plan" / "s2_5_repair_dispatch.json").read_text(encoding="utf-8"))
    assert dispatch["repair_lane"] == "s2_5_only_implementation_repair"
    assert dispatch["selected_candidate_id"] == "forward_probe_impl"
    assert dispatch["same_candidate_required"] is True
    assert dispatch["reuse_persistent_codex_session"] is True
    assert dispatch["activation_forward_probe_diagnostics"]["projector_output_identical"] is True
    assert dispatch["tensor_checks"]["identical_tensors"] == ["projector_output"]


def test_s3_full_metrics_failure_is_method_failure_even_with_implementation_signals(tmp_path: Path) -> None:
    project_root = tmp_path / "proj_full_metrics_method_failure"
    write_json(
        project_root / "experiment" / "results" / "main_results.json",
        {
            "baseline": {"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
            "acceptance": {"passed": False, "reason": "full metric below baseline"},
            "candidate_results": [
                {
                    "id": "proxy_pass_full_collapse",
                    "title": "Proxy pass full collapse",
                    "decision": "not_viable",
                    "command_status": "ok",
                    "mechanism_type": "utility_predicted_cache_routing",
                    "metrics": {"mean": 43.0, "datasets": {"mmlu-redux": 42.0}},
                    "delta_vs_baseline": -7.0,
                    "proxy_screen": {
                        "status": "passed",
                        "proxy_delta_vs_baseline": 0.6,
                        "proxy_dataset_deltas": {"mmlu-redux": 0.7},
                        "activation_smoke": {
                            "status": "failed",
                            "mechanism_trace": {"status": "missing"},
                        },
                        "proxy_effect_repair_contract": {"source": "proxy_activation_smoke"},
                    },
                    "activation_smoke": {
                        "status": "failed",
                        "mechanism_trace": {"status": "missing"},
                    },
                    "patch_result": {
                        "status": "applied",
                        "validation": {
                            "checks": [
                                {
                                    "name": "runtime_smoke:mechanism_activation_forward_probe",
                                    "returncode": 1,
                                    "failure_category": "mechanism_activation_forward_probe_failed",
                                    "probe": {
                                        "probe_type": "repo_small_batch_forward",
                                        "failures": ["enabled_disabled_forward_tensors_identical"],
                                    },
                                }
                            ]
                        },
                    },
                    "failure_attribution": {"primary_failure": "proxy_activation_smoke_no_effect"},
                }
            ],
        },
    )
    write_json(
        project_root / "experiment" / "results" / "proxy_calibration.json",
        {
            "current_iteration": {
                "proxy_false_positive_count": 1,
                "proxy_false_positive_rate": 1.0,
            }
        },
    )
    registry = {"iteration": 1}

    assert orchestrator_module._s3_feedback_failure_class(project_root) == "method_failure"
    feedback = orchestrator_module._s3_full_performance_feedback(
        project_root,
        registry,
        {"status": "not_viable"},
        "full metric below baseline",
    )

    assert feedback["summary"]["failure_class"] == "method_failure"
    assert feedback["summary"]["does_not_consume_same_direction_attempt"] is False
    assert feedback["summary"]["full_s3_completed_candidates"] == 1
    assert feedback["summary"]["proxy_false_positive_count"] == 1


def test_s3_full_metrics_repairable_does_not_use_implementation_repair_route(tmp_path: Path) -> None:
    project_root = tmp_path / "proj_full_metrics_repairable_method_route"
    write_json(
        project_root / "experiment" / "results" / "main_results.json",
        {
            "acceptance": {"passed": False, "reason": "full metric below baseline"},
            "candidate_results": [
                {
                    "id": "proxy_pass_full_collapse",
                    "decision": "proxy_repairable",
                    "command_status": "ok",
                    "metrics": {"mean": 43.0, "datasets": {"mmlu-redux": 42.0}},
                    "delta_vs_baseline": -7.0,
                    "proxy_screen": {
                        "status": "repairable_proxy_risk",
                        "proxy_delta_vs_baseline": 0.6,
                        "proxy_effect_repair_contract": {"source": "proxy_activation_smoke"},
                    },
                    "activation_smoke": {
                        "status": "failed",
                        "mechanism_trace": {"status": "missing"},
                    },
                    "failure_attribution": {"primary_failure": "proxy_activation_smoke_no_effect"},
                }
            ],
        },
    )
    config = {
        "orchestration": {
            "failure_feedback": {
                "enabled": True,
                "route_repairable_proxy_to_s2": True,
                "route_s3_failure_to_s1": True,
                "max_same_direction_proxy_failures": 5,
            }
        }
    }
    registry = {
        "iteration": 1,
        "max_iterations": 5,
        "status": "running",
        "blocked_reason": None,
        "current_stage": "S3_experiment",
        "repair_routes": {"1": 0},
        "implementation_repair_routes": {},
        "stages": {
            "S1_literature": {"status": "completed", "judge_retries": 0},
            "S2_plan": {"status": "completed", "judge_retries": 0},
            "S3_experiment": {"status": "running", "judge_retries": 0},
            "S4_writing": {"status": "pending", "judge_retries": 0},
            "S5_review": {"status": "pending", "judge_retries": 0},
        },
    }

    assert orchestrator_module._s3_feedback_failure_class(project_root) == "method_failure"
    assert Orchestrator._should_route_s3_implementation_failure_to_s2(config, project_root, registry) is False
    assert Orchestrator._should_route_s3_repairable_proxy_to_s2(config, project_root, registry) is True
    routed = Orchestrator._route_s3_repairable_proxy_to_s2(project_root, registry, {"status": "blocked"}, "full metric below baseline", config=config)

    assert routed["failure_class"] == "method_failure"
    assert routed["same_direction_failure_count"] == 1
    assert registry["implementation_repair_routes"] == {}
    assert registry["repair_routes"]["1"] == 1
    feedback = json.loads((project_root / "plan" / "performance_feedback.json").read_text(encoding="utf-8"))
    assert feedback["summary"]["failure_class"] == "method_failure"
    assert feedback["summary"]["does_not_consume_same_direction_attempt"] is False


def test_s3_proxy_baseline_blocked_is_not_implementation_failure(tmp_path: Path) -> None:
    project_root = tmp_path / "proj_proxy_baseline_blocked"
    write_json(
        project_root / "experiment" / "results" / "main_results.json",
        {
            "acceptance": {"passed": False, "reason": "no candidate metrics"},
            "candidate_results": [
                {
                    "id": "idea_proxy_timeout",
                    "decision": "blocked",
                    "command_status": "blocked",
                    "metrics": None,
                    "proxy_screen": {
                        "status": "baseline_blocked",
                        "reason": "proxy baseline eval mmlu-redux failed",
                        "baseline_status": "blocked",
                        "baseline_failure": {
                            "category": "proxy_timeout",
                            "step": "proxy_baseline_eval_mmlu-redux",
                            "returncode": 124,
                        },
                    },
                    "failure_attribution": {"primary_failure": "none"},
                }
            ],
        },
    )
    config = {
        "orchestration": {
            "failure_feedback": {
                "enabled": True,
                "route_repairable_proxy_to_s2": True,
                "route_implementation_failure_to_s2": True,
            }
        }
    }
    registry = {"iteration": 1, "implementation_repair_routes": {}}

    assert orchestrator_module._s3_feedback_failure_class(project_root) == "method_failure"
    assert Orchestrator._should_route_s3_implementation_failure_to_s2(config, project_root, registry) is False


def test_s3_proxy_rejected_with_metrics_routes_as_method_feedback(tmp_path: Path) -> None:
    project_root = tmp_path / "proj_proxy_rejected_metrics_method"
    write_json(
        project_root / "experiment" / "results" / "main_results.json",
        {
            "acceptance": {
                "passed": False,
                "reason": "proxy mean delta -2.703 below hard threshold -0.3",
            },
            "candidate_results": [
                {
                    "id": "inline_validmask_coverage_ablation_repair",
                    "decision": "proxy_rejected",
                    "command_status": "proxy_rejected",
                    "metrics": None,
                    "proxy_screen": {
                        "status": "rejected",
                        "reason": "proxy mean delta -2.703 below hard threshold -0.3",
                        "metrics": {
                            "mean": 36.6112,
                            "datasets": {
                                "ai2-arc": 39.6825,
                                "mmlu-redux": 34.2135,
                                "openbookqa": 35.9375,
                            },
                        },
                        "proxy_baseline": {
                            "mean": 39.3142,
                            "datasets": {
                                "ai2-arc": 38.0952,
                                "mmlu-redux": 37.6598,
                                "openbookqa": 42.1875,
                            },
                        },
                        "proxy_delta_vs_proxy_baseline": -2.703,
                        "proxy_dataset_deltas": {
                            "ai2-arc": 1.5873,
                            "mmlu-redux": -3.4463,
                            "openbookqa": -6.25,
                        },
                    },
                    "failure_attribution": {
                        "primary_failure": "cheap_proxy_rejected_before_full_training",
                        "patch_risk": {
                            "risk_labels": [
                                "alignment_mechanism_changed",
                                "config_override_changed",
                                "projector_mechanism_changed",
                                "test_change",
                                "training_loop_changed",
                            ]
                        },
                    },
                    "patch_result": {
                        "status": "snapshot_applied",
                        "changed_files": [
                            "rosetta/model/aligner.py",
                            "rosetta/model/projector.py",
                            "script/train/SFT_train.py",
                            "test/test_activation_forward_probe.py",
                        ],
                    },
                }
            ],
        },
    )
    config = {
        "orchestration": {
            "failure_feedback": {
                "enabled": True,
                "route_proxy_rejected_to_s2": True,
                "route_repairable_proxy_to_s2": True,
                "max_same_direction_proxy_failures": 5,
            }
        }
    }
    registry = {"iteration": 1, "proxy_rejected_routes": {}, "implementation_repair_routes": {}}

    assert orchestrator_module._s3_feedback_failure_class(project_root) == "method_failure"
    assert Orchestrator._should_route_s3_implementation_failure_to_s2(config, project_root, registry) is False
    assert Orchestrator._should_route_s3_proxy_rejected_to_s2(config, project_root, registry) is True


def test_s3_repairable_proxy_budget_returns_to_s1_on_final_failure(tmp_path: Path) -> None:
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
                "route_s3_failure_to_s1": True,
                "max_proxy_repair_routes_per_iteration": 3,
                "max_same_direction_proxy_failures": 5,
            }
        }
    }
    registry = {
        "iteration": 1,
        "max_iterations": 5,
        "status": "running",
        "blocked_reason": None,
        "current_stage": "S3_experiment",
        "repair_routes": {"1": 1},
        "stages": {
            "S1_literature": {"status": "completed", "judge_retries": 0},
            "S2_plan": {"status": "completed", "judge_retries": 0},
            "S3_experiment": {"status": "running", "judge_retries": 0},
            "S4_writing": {"status": "pending", "judge_retries": 0},
            "S5_review": {"status": "pending", "judge_retries": 0},
        },
    }

    assert Orchestrator._should_route_s3_repairable_proxy_to_s2(config, project_root, registry) is True
    assert Orchestrator._should_route_s3_repairable_proxy_to_s1(config, project_root, registry) is False
    registry["repair_routes"]["1"] = 2
    assert Orchestrator._should_route_s3_repairable_proxy_to_s2(config, project_root, registry) is True
    assert Orchestrator._should_route_s3_repairable_proxy_to_s1(config, project_root, registry) is False
    registry["repair_routes"]["1"] = 3
    assert Orchestrator._should_route_s3_repairable_proxy_to_s2(config, project_root, registry) is True
    assert Orchestrator._should_route_s3_repairable_proxy_to_s1(config, project_root, registry) is False
    registry["repair_routes"]["1"] = 4
    assert Orchestrator._should_route_s3_repairable_proxy_to_s2(config, project_root, registry) is False
    assert Orchestrator._should_route_s3_repairable_proxy_to_s1(config, project_root, registry) is True
    routed = Orchestrator._route_s3_repairable_proxy_to_s1(
        project_root,
        registry,
        {"status": "blocked"},
        "repairable proxy budget exhausted",
        config=config,
    )

    assert routed["status"] == "routed"
    assert routed["next_stage"] == "S1_literature"
    assert routed["same_direction_failure_count"] == 5
    assert routed["same_direction_failure_budget"] == 5
    assert registry["iteration"] == 2
    assert registry["current_stage"] == "S1_literature"
    feedback = json.loads((project_root / "plan" / "performance_feedback.json").read_text(encoding="utf-8"))
    assert feedback["summary"]["recommended_s2_action"] == "return_to_s1_new_direction"
    assert feedback["summary"]["s2_action_policy"]["matched_rule"] == "failure_budget_exhausted"


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
    assert feedback["summary"]["s2_action_policy"]["matched_rule"] == "single_dataset_small_drop"
    assert "single_dataset_small_drop" in feedback["summary"]["repair_vs_variant_signals"]
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

    assert feedback["summary"]["failure_class"] == "implementation_failure"
    assert feedback["summary"]["does_not_consume_same_direction_attempt"] is True
    assert feedback["summary"]["recommended_s2_action"] == "patch_repair"
    assert feedback["summary"]["s2_action_policy"]["matched_rule"] == "implementation_failure"
    assert "implementation_failure" in feedback["summary"]["repair_vs_variant_signals"]


def test_s2_5_validation_failure_routes_to_patch_only_repair(tmp_path: Path) -> None:
    project_root = tmp_path / "proj_s2_5_validation_repair"
    validation_path = project_root / "plan" / "code_patches" / "idea_impl" / "validation.json"
    write_json(
        validation_path,
        {
            "status": "validation_failed",
            "checks": [
                {"name": "py_compile:rosetta/model/projector.py", "returncode": 0},
                {
                    "name": "runtime_smoke:first_batch_train",
                    "returncode": 1,
                    "failure_category": "first_batch_runtime_error",
                    "repair_hint": "Fix the patched forward/train path so a one-sample first batch can complete before proxy train.",
                },
            ],
            "changed_files": ["rosetta/model/projector.py"],
        },
    )
    write_json(
        project_root / "plan" / "code_patches" / "patch_manifest.json",
        {
            "status": "no_valid_patch",
            "valid_patch_count": 0,
            "retryable_patch_count": 0,
            "selected_candidate_id": None,
            "candidates": [
                {
                    "candidate_id": "idea_impl",
                    "title": "Implementation Repair",
                    "status": "validation_failed",
                    "validation": "plan/code_patches/idea_impl/validation.json",
                    "changed_files": ["rosetta/model/projector.py"],
                    "variant_fingerprint": "abc123",
                    "reason": "validation_failed",
                }
            ],
        },
    )
    registry = {
        "iteration": 1,
        "status": "running",
        "current_stage": "S2_plan",
        "implementation_repair_routes": {},
        "stages": {
            "S1_literature": {"status": "completed", "judge_retries": 0},
            "S2_plan": {"status": "running", "judge_retries": 2},
            "S3_experiment": {"status": "pending", "judge_retries": 1},
            "S4_writing": {"status": "pending", "judge_retries": 0},
            "S5_review": {"status": "pending", "judge_retries": 0},
        },
    }
    config = {
        "orchestration": {
            "failure_feedback": {
                "enabled": True,
                "max_implementation_repair_routes_per_iteration": 4,
            }
        }
    }
    gate_report = SimpleNamespace(
        to_dict=lambda: {
            "checks": [
                {
                    "name": "s2_5_patch_manifest_status",
                    "status": "FAIL",
                    "message": "S2.5 patch manifest status is no_valid_patch",
                }
            ]
        }
    )

    routed = Orchestrator._route_s2_5_validation_failure_to_repair(
        project_root,
        registry,
        {"status": "running"},
        gate_report,
        "S2.5 patch manifest status is no_valid_patch",
        config=config,
    )

    assert routed is not None
    assert routed["status"] == "routed"
    assert routed["repair_lane"] == "s2_5_only_implementation_repair"
    assert routed["skips_s2_planner"] is True
    assert routed["does_not_consume_same_direction_attempt"] is True
    assert registry["implementation_repair_routes"]["1"] == 1
    assert registry["stages"]["S2_plan"]["judge_retries"] == 0
    assert registry["stages"]["S3_experiment"]["judge_retries"] == 0
    feedback = json.loads((project_root / "plan" / "performance_feedback.json").read_text(encoding="utf-8"))
    assert feedback["summary"]["failure_class"] == "implementation_failure"
    assert feedback["summary"]["s2_action_policy"]["skips_s2_planner"] is True
    assert feedback["candidate_results"][0]["runtime_validation"]["runtime_smoke"] == "failed"
    dispatch = json.loads((project_root / "plan" / "s2_5_repair_dispatch.json").read_text(encoding="utf-8"))
    assert dispatch["selected_candidate_id"] == "idea_impl"
    assert dispatch["variant_fingerprint"] == "abc123"
    assert dispatch["runtime_validation"]["runtime_smoke"] == "failed"
    assert dispatch["same_candidate_required"] is True
    assert dispatch["do_not_replan_method"] is True


def test_s2_5_validation_repair_budget_is_per_candidate_variant(tmp_path: Path) -> None:
    project_root = tmp_path / "proj_s2_5_per_candidate_budget"
    validation_path = project_root / "plan" / "code_patches" / "new_impl" / "validation.json"
    write_json(
        validation_path,
        {
            "status": "validation_failed",
            "checks": [
                {"name": "py_compile:rosetta/model/projector.py", "returncode": 0},
                {
                    "name": "runtime_smoke:mechanism_activation_wiring",
                    "returncode": 1,
                    "failure_category": "mechanism_activation_wiring_failed",
                    "stderr": "runtime model files mention ablation_disable_new_impl but no forward function reads it",
                },
            ],
            "changed_files": ["rosetta/model/projector.py"],
        },
    )
    write_json(
        project_root / "plan" / "code_patches" / "patch_manifest.json",
        {
            "status": "no_valid_patch",
            "valid_patch_count": 0,
            "retryable_patch_count": 0,
            "selected_candidate_id": None,
            "candidates": [
                {
                    "candidate_id": "new_impl",
                    "title": "New Implementation Repair",
                    "status": "validation_failed",
                    "validation": "plan/code_patches/new_impl/validation.json",
                    "changed_files": ["rosetta/model/projector.py"],
                    "variant_fingerprint": "new_fingerprint",
                    "reason": "validation_failed",
                }
            ],
        },
    )
    registry = {
        "iteration": 1,
        "status": "running",
        "current_stage": "S2_plan",
        "implementation_repair_routes": {"1": 8},
        "implementation_repair_routes_by_candidate": {"1:old_impl:old_fingerprint": 8},
        "stages": {
            "S1_literature": {"status": "completed", "judge_retries": 0},
            "S2_plan": {"status": "running", "judge_retries": 2},
            "S3_experiment": {"status": "pending", "judge_retries": 1},
            "S4_writing": {"status": "pending", "judge_retries": 0},
            "S5_review": {"status": "pending", "judge_retries": 0},
        },
    }
    config = {
        "orchestration": {
            "failure_feedback": {
                "enabled": True,
                "max_implementation_repair_routes_per_iteration": 8,
            }
        }
    }

    routed = Orchestrator._route_s2_5_validation_failure_to_repair(
        project_root,
        registry,
        {"status": "running"},
        SimpleNamespace(to_dict=lambda: {"checks": []}),
        "S2.5 patch manifest status is no_valid_patch",
        config=config,
    )

    assert routed is not None
    assert routed["repair_lane"] == "s2_5_only_implementation_repair"
    assert routed["repair_route_key"] == "1:new_impl:new_fingerprint"
    assert registry["implementation_repair_routes"]["1"] == 9
    assert registry["implementation_repair_routes_by_candidate"]["1:old_impl:old_fingerprint"] == 8
    assert registry["implementation_repair_routes_by_candidate"]["1:new_impl:new_fingerprint"] == 1
    dispatch = json.loads((project_root / "plan" / "s2_5_repair_dispatch.json").read_text(encoding="utf-8"))
    assert dispatch["selected_candidate_id"] == "new_impl"


def test_s2_5_preflight_refreshes_stale_repair_dispatch_before_s2_agent(tmp_path: Path) -> None:
    project_root = tmp_path / "proj_s2_5_preflight_refresh"
    validation_path = project_root / "plan" / "code_patches" / "new_impl" / "validation.json"
    write_json(
        validation_path,
        {
            "status": "validation_failed",
            "checks": [
                {
                    "name": "runtime_smoke:mechanism_activation_wiring",
                    "returncode": 1,
                    "failure_category": "mechanism_activation_wiring_failed",
                    "stderr": "no forward function reads ablation_disable_new_impl",
                }
            ],
            "changed_files": ["rosetta/model/projector.py"],
        },
    )
    write_json(
        project_root / "plan" / "code_patches" / "patch_manifest.json",
        {
            "status": "no_valid_patch",
            "valid_patch_count": 0,
            "retryable_patch_count": 0,
            "selected_candidate_id": None,
            "candidates": [
                {
                    "candidate_id": "new_impl",
                    "status": "validation_failed",
                    "validation": "plan/code_patches/new_impl/validation.json",
                    "changed_files": ["rosetta/model/projector.py"],
                    "variant_fingerprint": "new_fingerprint",
                }
            ],
        },
    )
    write_json(
        project_root / "plan" / "s2_5_repair_dispatch.json",
        {
            "mode": "s2_5_only_implementation_repair",
            "status": "active",
            "selected_candidate_id": "old_impl",
            "variant_fingerprint": "old_fingerprint",
        },
    )
    registry = {
        "iteration": 1,
        "status": "running",
        "current_stage": "S2_plan",
        "implementation_repair_routes": {"1": 8},
        "implementation_repair_routes_by_candidate": {"1:old_impl:old_fingerprint": 8},
        "stages": {
            "S1_literature": {"status": "completed", "judge_retries": 0},
            "S2_plan": {"status": "running", "judge_retries": 0},
            "S3_experiment": {"status": "pending", "judge_retries": 0},
            "S4_writing": {"status": "pending", "judge_retries": 0},
            "S5_review": {"status": "pending", "judge_retries": 0},
        },
    }
    config = {
        "orchestration": {
            "failure_feedback": {
                "enabled": True,
                "max_implementation_repair_routes_per_iteration": 8,
            }
        }
    }

    routed = Orchestrator._route_existing_s2_5_validation_failure_before_s2_agent(
        project_root,
        registry,
        config=config,
    )

    assert routed is not None
    assert routed["repair_route_key"] == "1:new_impl:new_fingerprint"
    dispatch = json.loads((project_root / "plan" / "s2_5_repair_dispatch.json").read_text(encoding="utf-8"))
    assert dispatch["selected_candidate_id"] == "new_impl"
    assert dispatch["variant_fingerprint"] == "new_fingerprint"
    assert registry["implementation_repair_routes_by_candidate"]["1:new_impl:new_fingerprint"] == 1


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

import json

from auto_research.cli import main
from auto_research.reporting import build_memory_report, build_project_report, format_memory_report, format_project_report
from auto_research.utils import write_json, write_yaml


def test_project_report_summarizes_c2c_route(tmp_path, monkeypatch, capsys) -> None:
    project = tmp_path / "proj_report"
    (project / "meta").mkdir(parents=True)
    (project / "literature").mkdir()
    (project / "plan").mkdir()
    (project / "experiment" / "results").mkdir(parents=True)
    write_yaml(
        project / "meta" / "registry.yaml",
        {
            "project_id": "proj_report",
            "research_topic": "cross tokenizer cache",
            "status": "running",
            "current_stage": "S2_plan",
            "iteration": 2,
            "blocked_reason": None,
            "stages": {},
        },
    )
    write_json(
        project / "literature" / "ideas.json",
        [
            {
                "id": "utility_predicted_cache_routing",
                "title": "Utility Predicted Cache Routing",
                "selected": True,
                "mechanism_type": "utility_predicted_cache_routing",
            }
        ],
    )
    write_json(
        project / "plan" / "direction_scorecard.json",
        {
            "current_direction": {
                "direction_id": "utility_predicted_cache_routing",
                "title": "Utility Predicted Cache Routing",
                "mechanism_type": "utility_predicted_cache_routing",
                "summary": {
                    "same_direction_failure_count": 2,
                    "same_direction_failure_budget": 5,
                    "best_proxy_delta": 0.4,
                    "direction_quality": "mixed_direction_evidence",
                },
                "attempts": [
                    {
                        "iteration": 1,
                        "route": "proxy_rejected_same_direction",
                        "candidate_ids": ["utility_v1"],
                        "best_proxy_delta": -1.2,
                        "dragging_datasets": [{"dataset": "mmlu-redux"}],
                    }
                ],
            }
        },
    )
    write_json(
        project / "plan" / "performance_feedback.json",
        {
            "reason": "proxy rejected",
            "summary": {
                "recommended_s2_action": "mechanism_repair",
                "next_action": "repair_or_variant_same_direction",
                "s2_action_policy": {
                    "action": "mechanism_repair",
                    "matched_rule": "single_dataset_small_drop",
                    "reason": "only one dataset dropped",
                },
                "repair_vs_variant_signals": ["single_dataset_small_drop"],
            },
        },
    )
    write_json(
        project / "experiment" / "results" / "main_results.json",
        {
            "best_proxy_candidate": {
                "id": "utility_v2",
                "title": "Utility v2",
                "decision": "proxy_rejected",
                "delta_vs_baseline": None,
                "proxy_screen": {"proxy_delta_vs_baseline": 0.4, "proxy_dataset_deltas": {"mmlu-redux": -0.2, "ai2-arc": 0.5}},
                "patch_result": {"changed_files": ["rosetta/model/aligner.py"]},
            },
            "candidate_results": [],
        },
    )
    write_json(
        project / "meta" / "route_decision.json",
        {
            "schema_version": "c2c_route_decision_v1",
            "created_at": "2026-07-03T00:00:00Z",
            "trigger_stage": "S3_experiment",
            "trigger_source": "s3_gate",
            "failure_class": "proxy_negative",
            "decision": "route_to_s2",
            "next_stage": "S2_plan",
            "reason_codes": ["proxy_decision_report_route_hint_return_s2"],
            "budget_effects": {
                "consumes_same_direction_attempt": True,
                "consumes_patch_repair_attempt": False,
                "consumes_resource_retry": False,
                "increments_iteration": False,
            },
            "memory_effects": {"write_shared_method_memory": True, "memory_kind": "proxy_rejected_variant"},
            "artifact_effects": {
                "invalidate_from": "S2_plan",
                "preserve_s1_direction": True,
                "preserve_s2_selected_variant": False,
                "preserve_s2_5_patch_lock": False,
            },
            "orchestrator_action": {"registry_current_stage": "S2_plan", "status": "feedback_routed"},
        },
    )
    write_json(
        project / "meta" / "attempt_ledger.json",
        {
            "schema_version": "c2c_attempt_ledger_v1",
            "project_id": "proj_report",
            "records": [],
            "counters": {"by_direction": {"utility_predicted_cache_routing": {"proxy_failures": 1, "full_s3_failures": 0, "patch_repairs": 0, "resource_retries": 0}}},
        },
    )
    write_json(project / "literature" / "c2c" / "evidence_quality_score.json", {"gate": "pass", "novelty_score": 0.68})
    write_json(project / "plan" / "s2_planner" / "planner_gate_report.json", {"gate": "pass", "selected_variant_id": "utility_v2"})
    write_json(project / "plan" / "s2_planner" / "variant_scorecard.json", {"ranking": [{"variant_id": "utility_v2", "score": 0.72, "decision": "selected"}]})
    write_json(project / "experiment" / "results" / "c2c_proxy_decision_report.json", {"decision": "proxy_rejected", "route_hint": "return_s2", "failure_class": "proxy_negative"})
    write_json(project / "meta" / "c2c_e2e_readiness_report.json", {"gate": "pass"})
    write_json(project / "meta" / "c2c_artifact_audit_report.json", {"gate": "fail", "summary": {"missing": 1}})
    write_json(project / "meta" / "c2c_e2e_run_manifest.json", {"mode": "real", "final_status": "blocked"})
    write_json(project / "meta" / "c2c_replay_result.json", {"status": "match", "mismatches": []})
    (project / "plan" / "code_patches").mkdir()
    write_json(
        project / "plan" / "code_patches" / "patch_manifest.json",
        {
            "status": "ok",
            "selected_candidate_id": "utility_patch_manifest",
            "valid_patch_ids": ["utility_patch_manifest"],
            "selected_patch": {
                "candidate_id": "utility_patch_manifest",
                "title": "Utility patch manifest",
                "status": "ok",
                "patch_json": "plan/code_patches/utility_patch_manifest/patch.json",
                "selected_variant": 2,
                "changed_files": ["rosetta/model/projector.py"],
                "quality_score": {"score": 80},
            },
        },
    )

    report = build_project_report(project)
    text = format_project_report(report)

    assert report["s1_direction"]["direction_id"] == "utility_predicted_cache_routing"
    assert report["same_direction_attempt"]["count"] == 2
    assert report["current_best_patch"]["candidate_id"] == "utility_v2"
    assert "stage_states" in report
    assert report["next_route"]["action"] == "route_to_s2"
    assert report["route"]["last_decision"] == "route_to_s2"
    assert report["s1_quality"]["evidence_gate"] == "pass"
    assert report["s2_planner"]["selected_variant_score"] == 0.72
    assert report["s3_proxy"]["route_hint"] == "return_s2"
    assert report["attempt_ledger"]["same_direction_proxy_failures"] == 1
    assert report["e2e"]["readiness_gate"] == "pass"
    assert report["e2e"]["artifact_audit_gate"] == "fail"
    assert report["e2e"]["real_run_manifest"]["mode"] == "real"
    assert report["e2e"]["replay"]["last_replay_status"] == "match"
    assert "S1 direction: utility_predicted_cache_routing" in text
    assert "Stage states:" in text
    assert "Next route: route_to_s2" in text
    assert "Route decision: route_to_s2 -> S2_plan" in text
    assert "C2C E2E: readiness=pass audit=fail replay=match" in text

    import auto_research.config as config_module
    import auto_research.orchestrator as orchestrator_module

    config = {"project": {"workspace_root": str(tmp_path)}}
    monkeypatch.setattr(config_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)
    main(["report", "--project-id", "proj_report"])
    output = capsys.readouterr().out
    assert "Same-direction attempt: 2/5" in output

    main(["report", "--project-id", "proj_report", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["next_route"]["matched_rule"] == "proxy_decision_report_route_hint_return_s2"


def test_memory_report_summarizes_shared_pool_and_project_retrieval(tmp_path, monkeypatch, capsys) -> None:
    memory_path = tmp_path / "method_memory.jsonl"
    summary_path = tmp_path / "method_memory.md"
    entries = [
        {
            "schema_version": "shared_method_failure_memory_v1",
            "memory_id": "mem_proxy_false_positive",
            "timestamp": "2026-06-10T00:00:00Z",
            "project_id": "old_project",
            "route": "full_s3_failure",
            "summary": {
                "dataset_regressions": {"mmlu-redux": 1.5},
                "dragging_datasets": [{"dataset": "mmlu-redux", "regression": 1.5}],
            },
            "memory_quality": {"priority": 12.5, "signals": ["proxy_full_false_positive", "proxy_dataset_misprediction"]},
            "proxy_calibration": {
                "summary": {
                    "proxy_false_positive_count": 1,
                    "proxy_false_positive_rate": 1.0,
                    "dataset_error_summary": {"mmlu-redux": {"misprediction_count": 1, "max_abs_proxy_full_delta_error": 2.4}},
                    "mechanism_false_positive_summary": {
                        "utility_predicted_cache_routing": {"count": 1, "false_positive_count": 1}
                    },
                },
                "false_positive_candidates": [{"id": "proxy_pass_full_fail", "mechanism_type": "utility_predicted_cache_routing"}],
            },
        },
        {
            "schema_version": "shared_method_failure_memory_v1",
            "memory_id": "mem_regular_fail",
            "timestamp": "2026-06-09T00:00:00Z",
            "project_id": "old_project_2",
            "route": "proxy_rejected_same_direction",
            "summary": {"dataset_regressions": {"ai2-arc": 0.8}},
            "memory_quality": {"priority": 3.0, "signals": ["dataset_regression"]},
            "direction_scorecard": {
                "mechanism_type": "semantic_span_graph_alignment",
                "summary": {"best_proxy_delta": -0.4, "direction_quality": "mixed_direction_evidence"},
            },
        },
    ]
    memory_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in entries) + "\n", encoding="utf-8")
    project = tmp_path / "proj_memory"
    (project / "meta").mkdir(parents=True)
    write_yaml(
        project / "meta" / "project_config.yaml",
        {
            "orchestration": {
                "shared_method_memory": {
                    "enabled": True,
                    "path": str(memory_path),
                    "summary_path": str(summary_path),
                    "prompt_limit": 1,
                }
            }
        },
    )
    write_yaml(
        project / "meta" / "registry.yaml",
        {"project_id": "proj_memory", "research_topic": "topic", "status": "running", "current_stage": "S1_literature", "iteration": 1, "stages": {}},
    )
    config = {
        "project": {"workspace_root": str(tmp_path)},
        "orchestration": {"shared_method_memory": {"enabled": True, "path": str(memory_path), "summary_path": str(summary_path), "prompt_limit": 1}},
    }

    report = build_memory_report(config=config, project_root=project)
    text = format_memory_report(report)

    assert report["method_failure_count"] == 2
    assert report["top_failed_mechanisms"][0]["mechanism_type"] == "utility_predicted_cache_routing"
    assert report["top_dragging_datasets"][0]["dataset"] == "mmlu-redux"
    assert report["recent_memory"][0]["memory_id"] == "mem_proxy_false_positive"
    assert report["project_retrieval"]["memory_ids"] == ["mem_proxy_false_positive"]
    assert "Shared Method Memory" in text
    assert "Current project retrieval: proj_memory" in text

    import auto_research.config as config_module
    import auto_research.orchestrator as orchestrator_module

    monkeypatch.setattr(config_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)
    main(["memory", "report", "--project-id", "proj_memory"])
    output = capsys.readouterr().out
    assert "Method failures: 2" in output
    assert "mem_proxy_false_positive" in output

    main(["memory", "report", "--project-id", "proj_memory", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["project_retrieval"]["memory_ids"] == ["mem_proxy_false_positive"]


def test_project_report_surfaces_retryable_pause_resume_instruction(tmp_path) -> None:
    project = tmp_path / "proj_retryable_report"
    (project / "meta").mkdir(parents=True)
    (project / "literature").mkdir()
    (project / "plan" / "code_patches").mkdir(parents=True)
    (project / "experiment" / "results").mkdir(parents=True)
    write_yaml(
        project / "meta" / "registry.yaml",
        {
            "project_id": "proj_retryable_report",
            "research_topic": "cross tokenizer cache",
            "status": "retryable_paused",
            "current_stage": "S2_plan",
            "iteration": 1,
            "blocked_reason": "S2.5 patch generation hit a retryable Codex/backend quota or rate-limit failure.",
            "pause_type": "codex_quota_or_rate_limit",
            "resume_instruction": "Wait for quota/rate limit recovery, then run auto-research resume --project-id proj_retryable_report",
            "stages": {},
        },
    )
    write_json(project / "literature" / "ideas.json", [])
    write_json(
        project / "plan" / "code_patches" / "patch_manifest.json",
        {"status": "retryable_no_valid_patch", "retryable_patch_count": 1, "retryable": True},
    )

    report = build_project_report(project)
    text = format_project_report(report)

    assert report["status"] == "retryable_paused"
    assert report["next_route"]["action"] == "resume_after_quota_recovery"
    assert report["next_route"]["matched_rule"] == "codex_quota_or_rate_limit"
    assert "Next route: resume_after_quota_recovery" in text
    assert "Pause type: codex_quota_or_rate_limit" in text
    assert "auto-research resume --project-id proj_retryable_report" in text


def test_project_report_surfaces_full_s3_readiness_states(tmp_path) -> None:
    project = tmp_path / "proj_report_readiness"
    (project / "meta").mkdir(parents=True)
    (project / "literature").mkdir()
    (project / "plan").mkdir()
    (project / "experiment" / "results").mkdir(parents=True)
    write_yaml(
        project / "meta" / "registry.yaml",
        {
            "project_id": "proj_report_readiness",
            "research_topic": "cross tokenizer cache",
            "status": "running",
            "current_stage": "S3_experiment",
            "iteration": 3,
            "stages": {},
        },
    )
    write_json(
        project / "literature" / "ideas.json",
        [{"id": "utility_predicted_cache_routing", "title": "Utility", "selected": True}],
    )
    write_json(
        project / "experiment" / "results" / "full_s3_readiness_report.json",
        {
            "schema_version": "c2c_proxy_to_full_readiness_v1",
            "candidate_id": "utility_v3",
            "status": "ready",
            "full_train_allowed": True,
            "proxy": {"status": "passed", "delta": 0.8},
            "activation_smoke": {"status": "passed", "no_op": False},
            "eval_smoke": {"status": "ok", "healthy": True},
            "ablation_switch": {"declared": True, "switch": "disable_utility"},
            "worth_full_train": {"decision": "yes", "reason": "cheap proxy passed"},
        },
    )
    write_json(
        project / "experiment" / "results" / "main_results.json",
        {
            "best_proxy_candidate": {
                "id": "utility_v3",
                "decision": "proxy_passed",
                "proxy_screen": {"status": "passed", "proxy_delta_vs_baseline": 0.8},
                "full_s3_readiness": {
                    "status": "ready",
                    "full_train_allowed": True,
                    "activation_smoke": {"status": "passed", "no_op": False},
                    "proxy": {"status": "passed"},
                },
            },
            "candidate_results": [],
            "ablation_summary": {"status": "pending"},
        },
    )

    report = build_project_report(project)
    text = format_project_report(report)

    assert report["stage_states"]["proxy"] == "passed"
    assert report["stage_states"]["activation"] == "passed"
    assert report["stage_states"]["activation_no_op"] is False
    assert report["stage_states"]["full"] == "ready_or_running"
    assert report["stage_states"]["ablation"] == "pending"
    assert "Activation no-op: no" in text
    assert "Stage states: proxy=passed activation=passed full=ready_or_running ablation=pending" in text


def test_project_report_uses_selected_patch_manifest_when_no_result_candidate(tmp_path) -> None:
    project = tmp_path / "proj_report_manifest_selected"
    (project / "meta").mkdir(parents=True)
    (project / "literature").mkdir()
    (project / "plan" / "code_patches").mkdir(parents=True)
    (project / "experiment" / "results").mkdir(parents=True)
    write_yaml(
        project / "meta" / "registry.yaml",
        {
            "project_id": "proj_report_manifest_selected",
            "research_topic": "cross tokenizer cache",
            "status": "running",
            "current_stage": "S3_experiment",
            "iteration": 1,
            "stages": {},
        },
    )
    write_json(project / "literature" / "ideas.json", [])
    write_json(project / "experiment" / "results" / "main_results.json", {"candidate_results": []})
    write_json(
        project / "plan" / "code_patches" / "patch_manifest.json",
        {
            "status": "ok",
            "selected_candidate_id": "manifest_winner",
            "selected_patch": {
                "candidate_id": "manifest_winner",
                "title": "Manifest Winner",
                "status": "ok",
                "patch_json": "plan/code_patches/manifest_winner/patch.json",
                "selected_variant": 1,
                "changed_files": ["rosetta/model/aligner.py"],
                "quality_score": {"score": 72},
            },
        },
    )

    report = build_project_report(project)

    assert report["current_best_patch"]["candidate_id"] == "manifest_winner"
    assert report["current_best_patch"]["patch_json"] == "plan/code_patches/manifest_winner/patch.json"
    assert report["current_best_patch"]["selected_variant"] == 1


def test_project_report_prefers_current_s1_idea_over_previous_scorecard(tmp_path) -> None:
    project = tmp_path / "proj_report_new_direction"
    (project / "meta").mkdir(parents=True)
    (project / "literature").mkdir()
    (project / "plan").mkdir()
    (project / "experiment" / "results").mkdir(parents=True)
    write_yaml(
        project / "meta" / "registry.yaml",
        {
            "project_id": "proj_report_new_direction",
            "research_topic": "cross tokenizer cache",
            "status": "running",
            "current_stage": "S2_plan",
            "iteration": 2,
            "blocked_reason": None,
            "stages": {},
        },
    )
    write_json(
        project / "literature" / "ideas.json",
        [
            {
                "id": "pathology_conditioned_transfer_controller",
                "title": "Pathology-Conditioned Transfer Controller",
                "selected": True,
                "mechanism_type": "pathology_conditioned_controller",
            }
        ],
    )
    write_json(
        project / "plan" / "direction_scorecard.json",
        {
            "current_direction": {
                "direction_id": "candidate_constrained_soft_span_alignment",
                "title": "Candidate-Constrained Soft Span Alignment",
                "mechanism_type": "semantic_span_graph_alignment",
                "summary": {
                    "same_direction_failure_count": 4,
                    "same_direction_failure_budget": 4,
                    "best_proxy_delta": -0.085,
                    "direction_quality": "mixed_direction_evidence",
                },
                "attempts": [{"iteration": 1, "route": "repairable_proxy_risk", "candidate_ids": ["candidate_constrained_soft_span_alignment"]}],
            }
        },
    )

    report = build_project_report(project)

    assert report["s1_direction"]["direction_id"] == "pathology_conditioned_transfer_controller"
    assert report["s1_direction"]["previous_direction_id"] == "candidate_constrained_soft_span_alignment"
    assert report["same_direction_attempt"]["count"] == 4

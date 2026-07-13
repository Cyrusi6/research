import json

from auto_research.cli import main
from auto_research.reporting import build_memory_report, build_project_report, format_memory_report, format_project_report
from auto_research.utils import write_json, write_yaml


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




def test_project_report_uses_canonical_route_and_research_state(tmp_path) -> None:
    from test_authoritative_state_machine import _complete, _direction, _initialize, _reserve, _variant
    from auto_research.research_state import ResearchEventLedger

    project = tmp_path / "proj_report_v2"
    (project / "meta").mkdir(parents=True)
    write_yaml(
        project / "meta" / "registry.yaml",
        {
            "project_id": project.name,
            "research_topic": "topic",
            "status": "running",
            "current_stage": "S2_plan",
            "iteration": 1,
            "stages": {},
        },
    )
    direction = _direction()
    variant = _variant(direction, 1)
    (project / "literature").mkdir()
    (project / "plan").mkdir()
    write_json(project / "literature" / "direction.json", direction)
    write_json(project / "plan" / "variant.json", variant)
    ledger = ResearchEventLedger(project)
    _initialize(ledger, direction, variant)
    _complete(ledger, _reserve(ledger, direction, variant), outcome="rejected")

    report = build_project_report(project)

    assert report["route"]["last_decision"] == "PROPOSE_NEXT_VARIANT"
    assert report["attempt_ledger"]["consumed"] == 1
    assert report["artifact_paths"]["route_outcome"] == "meta/route_outcome.json"
    assert report["artifact_paths"]["research_state"] == "meta/research_state.json"

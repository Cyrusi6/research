import json
from pathlib import Path

import auto_research.failure_log as failure_log_module
from auto_research.failure_log import FailureLogManager, build_c2c_feedback_bundle, load_c2c_feedback_bundle
from auto_research.method_memory import append_shared_c2c_method_failure, collect_used_shared_memory_refs, load_shared_method_memory, shared_method_memory_for_prompt


def test_failure_log_manager_records_and_cleans_runs(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    failed_run = runs_root / "bad_run"
    failed_run.mkdir(parents=True)
    (failed_run / "train.log").write_text("dummy\n", encoding="utf-8")

    manager = FailureLogManager(
        {"experiment": {"failure_log_filename": "failure.md"}},
        external_root=runs_root,
    )
    removed = manager.record_not_viable_ideas(
        project_id="proj_x",
        baseline_metrics={"rsum": 10, "t2i": {"R@1": 5}, "similarity_time": 1.0},
        candidate_results=[
            {
                "id": "idea_bad",
                "title": "Bad Idea",
                "direction": "test",
                "decision": "not_viable",
                "metrics": {"rsum": 8, "t2i": {"R@1": 4}, "similarity_time": 1.3},
                "train_log": str(failed_run / "train.log"),
            }
        ],
        cleanup=True,
    )

    assert removed == [str(failed_run)]
    assert not failed_run.exists()
    assert (runs_root / "failure.md").exists()
    assert (runs_root / "failure.jsonl").exists()
    payload = [json.loads(line) for line in (runs_root / "failure.jsonl").read_text(encoding="utf-8").splitlines()]
    assert payload[0]["idea_id"] == "idea_bad"


def test_c2c_feedback_bundle_summarizes_proxy_screen_deltas() -> None:
    bundle = build_c2c_feedback_bundle(
        [
            {
                "kind": "c2c_posthoc_feedback",
                "iteration": 1,
                "idea_id": "proxy_tradeoff",
                "decision": "proxy_repairable",
                "failure_mode": "proxy_repairable",
                "proxy_screen": {
                    "status": "repairable_proxy_risk",
                    "proxy_dataset_deltas": {"ai2-arc": -1.3, "openbookqa": 1.8},
                    "command_failure": {"category": "proxy_timeout"},
                },
            }
        ],
        project_id="proj",
        iteration=1,
    )

    summary = bundle["summary"]
    assert summary["failed_idea_ids"] == ["proxy_tradeoff"]
    assert summary["dataset_regressions"]["ai2-arc"] == 1.3
    assert summary["dragging_datasets"][0]["dataset"] == "ai2-arc"
    assert "science_reasoning_challenge" in summary["sample_type_failures"]
    assert "proxy_command_proxy_timeout" in summary["failure_modes"]


def test_c2c_method_feedback_view_drops_implementation_only_failures() -> None:
    entry = {
        "kind": "c2c_posthoc_feedback",
        "iteration": 4,
        "idea_id": "dtype_patch",
        "title": "Dtype Patch",
        "decision": "proxy_repairable",
        "failure_mode": "proxy_repairable",
        "reason": "proxy command 1 failed: RuntimeError expected scalar type Float but found BFloat16",
        "proxy_screen": {
            "status": "repairable_proxy_risk",
            "reason": "proxy command 1 failed: RuntimeError expected scalar type Float but found BFloat16",
            "command_failure": {
                "category": "dtype_mismatch",
                "summary": "RuntimeError expected scalar type Float but found BFloat16",
            },
        },
        "failure_attribution": {
            "primary_failure": "repairable_proxy_risk_before_full_training",
            "patch_risk": {
                "risk_labels": ["evaluation_code_changed"],
                "risk_files": [{"path": "script/evaluation/run_eval.py"}],
            },
        },
        "avoid_repeat_rule": "Do not rerun dtype_patch until proxy command failure dtype_mismatch is fixed.",
    }

    method_bundle = build_c2c_feedback_bundle([entry], project_id="proj", iteration=4, view="method")
    implementation_bundle = build_c2c_feedback_bundle([entry], project_id="proj", iteration=4, view="implementation")

    assert method_bundle["entries"] == []
    assert method_bundle["summary"]["failed_idea_ids"] == []
    assert method_bundle["summary"]["failure_modes"] == []
    assert method_bundle["summary"]["patch_risk_labels"] == []
    assert implementation_bundle["entries"][0]["proxy_screen"]["command_failure"]["category"] == "dtype_mismatch"
    assert "proxy_command_dtype_mismatch" in implementation_bundle["summary"]["failure_modes"]
    assert implementation_bundle["summary"]["patch_risk_labels"] == ["evaluation_code_changed"]


def test_c2c_method_feedback_view_keeps_metric_evidence_without_patch_noise() -> None:
    entry = {
        "kind": "c2c_posthoc_feedback",
        "iteration": 4,
        "idea_id": "proxy_tradeoff",
        "title": "Proxy Tradeoff",
        "decision": "proxy_rejected",
        "failure_mode": "proxy_rejected",
        "reason": "proxy mean delta -2.3 below hard threshold 0.0",
        "proxy_screen": {
            "status": "rejected",
            "reason": "proxy mean delta -2.3 below hard threshold 0.0",
            "metrics": {"mean": 47.7, "datasets": {"mmlu-redux": 46.0, "openbookqa": 51.0}},
            "baseline_metrics": {"mean": 50.0, "datasets": {"mmlu-redux": 50.0, "openbookqa": 50.0}},
            "proxy_delta_vs_baseline": -2.3,
            "proxy_dataset_deltas": {"mmlu-redux": -4.0, "openbookqa": 1.0},
            "command_failure": {"category": "dtype_mismatch"},
            "patch_risk": {"risk_labels": ["projector_mechanism_changed"]},
        },
        "failure_attribution": {
            "dragging_datasets": [
                {"dataset": "mmlu-redux", "sample_family": "multi_domain_knowledge_reasoning", "regression": 4.0}
            ],
            "patch_risk": {"risk_labels": ["projector_mechanism_changed"]},
        },
    }

    bundle = build_c2c_feedback_bundle([entry], project_id="proj", iteration=4, view="method")

    assert bundle["summary"]["failed_idea_ids"] == ["proxy_tradeoff"]
    assert bundle["summary"]["dataset_regressions"]["mmlu-redux"] == 4.0
    assert bundle["summary"]["patch_risk_labels"] == []
    assert "proxy_command_dtype_mismatch" not in bundle["summary"]["failure_modes"]
    assert "command_failure" not in bundle["entries"][0]["proxy_screen"]
    assert "patch_risk" not in bundle["entries"][0]["proxy_screen"]
    assert "patch_risk" not in bundle["entries"][0]["failure_attribution"]


def test_c2c_feedback_loader_splits_method_and_implementation_views(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    feedback_dir = project_root / "literature" / "feedback"
    feedback_dir.mkdir(parents=True)
    round_payload = {
        "summary_entry": {
            "kind": "c2c_feedback_summary",
            "iteration": 3,
            "failed_idea_ids": ["dtype_patch"],
            "summary_text": "proxy command dtype mismatch",
        },
        "entries": [
            {
                "kind": "c2c_posthoc_feedback",
                "iteration": 3,
                "idea_id": "dtype_patch",
                "title": "Dtype Patch",
                "decision": "proxy_repairable",
                "failure_mode": "proxy_repairable",
                "reason": "proxy command failed: dtype mismatch",
                "proxy_screen": {
                    "status": "repairable_proxy_risk",
                    "command_failure": {"category": "dtype_mismatch"},
                },
            },
            {
                "kind": "c2c_posthoc_feedback",
                "iteration": 3,
                "idea_id": "metric_tradeoff",
                "title": "Metric Tradeoff",
                "decision": "proxy_rejected",
                "failure_mode": "proxy_rejected",
                "reason": "proxy dataset regression mmlu-redux=3.0 exceeds hard threshold",
                "proxy_screen": {
                    "status": "rejected",
                    "proxy_dataset_deltas": {"mmlu-redux": -3.0, "openbookqa": 1.2},
                    "proxy_delta_vs_baseline": -0.4,
                },
            },
        ],
    }
    (feedback_dir / "failed_ideas_round_003.json").write_text(json.dumps(round_payload), encoding="utf-8")

    method_bundle = load_c2c_feedback_bundle(project_root, view="method")
    implementation_bundle = load_c2c_feedback_bundle(project_root, view="implementation")

    assert method_bundle["summary"]["failed_idea_ids"] == ["metric_tradeoff"]
    assert method_bundle["summary"]["dataset_regressions"]["mmlu-redux"] == 3.0
    assert "dtype_patch" not in method_bundle["summary"]["failed_idea_ids"]
    assert implementation_bundle["summary"]["failed_idea_ids"] == ["dtype_patch", "metric_tradeoff"]
    assert "proxy_command_dtype_mismatch" in implementation_bundle["summary"]["failure_modes"]


def test_c2c_feedback_loader_filters_retryable_resource_pause_noise(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    feedback_dir = project_root / "literature" / "feedback"
    results_dir = project_root / "experiment" / "results"
    feedback_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    resource_candidate = {
        "id": "gpu_wait_candidate",
        "title": "GPU Wait Candidate",
        "decision": "blocked",
        "failure_mode": "no_metrics",
        "proxy_screen": {
            "status": "resource_retry",
            "resource_retry": True,
            "failure_category": "s3_proxy_gpu_resource_retry",
            "reason": "cheap proxy resource wait timed out without an available GPU",
        },
    }
    method_candidate = {
        "id": "metric_tradeoff",
        "title": "Metric Tradeoff",
        "decision": "proxy_rejected",
        "failure_mode": "proxy_rejected",
        "proxy_screen": {
            "status": "rejected",
            "proxy_dataset_deltas": {"mmlu-redux": -2.0, "openbookqa": 0.5},
            "proxy_delta_vs_baseline": -0.4,
        },
    }
    (results_dir / "failure_feedback.json").write_text(
        json.dumps(
            {
                "entry": resource_candidate,
                "candidate_results": [resource_candidate, method_candidate],
                "failed_idea_ids": ["gpu_wait_candidate", "metric_tradeoff"],
                "avoid_repeat_rules": ["Do not rerun without fixing preflight, checkpoint, or evaluator failures."],
            }
        ),
        encoding="utf-8",
    )
    (feedback_dir / "failed_ideas_round_001.json").write_text(
        json.dumps({"entries": [resource_candidate]}),
        encoding="utf-8",
    )

    method_bundle = load_c2c_feedback_bundle(project_root, view="method")
    implementation_bundle = load_c2c_feedback_bundle(project_root, view="implementation")

    assert method_bundle["summary"]["failed_idea_ids"] == ["metric_tradeoff"]
    assert method_bundle["summary"]["dataset_regressions"]["mmlu-redux"] == 2.0
    assert "gpu_wait_candidate" not in json.dumps(method_bundle, ensure_ascii=False)
    assert "gpu_wait_candidate" not in json.dumps(implementation_bundle, ensure_ascii=False)
    assert "no_metrics" not in implementation_bundle["summary"]["failure_modes"]


def test_c2c_feedback_loader_uses_ledger_and_ignores_mutable_main_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "proj"
    ledger_path = project_root / "meta" / "research_events.sqlite3"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_bytes(b"authoritative-ledger-placeholder")
    results_dir = project_root / "experiment" / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "main_results.json").write_text(
        json.dumps(
            {
                "candidate_results": [
                    {
                        "id": "mutable-forgery",
                        "decision": "proxy_rejected",
                        "proxy_screen": {"proxy_dataset_deltas": {"mmlu-redux": -99.0}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class StubLedger:
        def __init__(self, root: Path):
            assert root == project_root

        def state(self) -> dict:
            return {
                "attempts": {"attempt-ledger": {"variant_id": "ledger-variant"}},
                "trial_results": {},
                "proxy_outcomes": {
                    "attempt-ledger": {
                        "attempt_id": "attempt-ledger",
                        "decision": "PROPOSE_NEXT_VARIANT",
                        "observed_delta": -0.4,
                        "dataset_deltas": {"mmlu-redux": -1.5, "openbookqa": 0.7},
                        "worst_dataset_regression": 1.5,
                        "reason_codes": ["proxy_contract_constraints_fail"],
                        "evidence_set_hash": "a" * 64,
                    }
                },
                "operation_events": {},
            }

    monkeypatch.setattr(failure_log_module, "ResearchEventLedger", StubLedger)

    bundle = load_c2c_feedback_bundle(project_root, view="method")

    assert bundle["summary"]["failed_idea_ids"] == ["ledger-variant"]
    assert bundle["summary"]["dataset_regressions"]["mmlu-redux"] == 1.5
    assert "mutable-forgery" not in json.dumps(bundle, ensure_ascii=False)
    assert "meta/research_events.sqlite3" in bundle["sources"]
    assert "experiment/results/main_results.json" not in bundle["sources"]


def test_shared_method_memory_records_only_method_failures(tmp_path: Path) -> None:
    config = {
        "orchestration": {
            "shared_method_memory": {
                "enabled": True,
                "path": str(tmp_path / "method_memory.jsonl"),
                "summary_path": str(tmp_path / "method_memory.md"),
            }
        }
    }
    project_root = tmp_path / "proj"
    (project_root / "meta").mkdir(parents=True)
    (project_root / "meta" / "registry.yaml").write_text("research_topic: c2c\n", encoding="utf-8")
    implementation_feedback = {
        "summary": {
            "failure_class": "implementation_failure",
            "does_not_consume_same_direction_attempt": True,
            "route": "implementation_failure",
        },
        "candidate_results": [
            {
                "id": "dtype_patch",
                "decision": "proxy_repairable",
                "reason": "RuntimeError dtype mismatch",
            }
        ],
    }
    method_feedback = {
        "summary": {
            "failure_class": "method_failure",
            "route": "proxy_rejected_same_direction",
            "iteration": 2,
        },
        "candidate_results": [
            {
                "id": "metric_tradeoff",
                "title": "Metric Tradeoff",
                "decision": "proxy_rejected",
                "proxy_screen": {
                    "status": "rejected",
                    "proxy_dataset_deltas": {"mmlu-redux": -2.5, "openbookqa": 1.0},
                    "proxy_delta_vs_baseline": -0.6,
                },
                "failure_attribution": {
                    "dragging_datasets": [
                        {"dataset": "mmlu-redux", "sample_family": "multi_domain_knowledge_reasoning", "regression": 2.5}
                    ]
                },
            }
        ],
    }

    skipped = append_shared_c2c_method_failure(config, project_root=project_root, performance_feedback=implementation_feedback)
    appended = append_shared_c2c_method_failure(config, project_root=project_root, performance_feedback=method_feedback)
    memory = load_shared_method_memory(config)

    assert skipped["status"] == "skipped"
    assert appended["status"] == "appended"
    assert memory["entry_count"] == 1
    assert memory["entries"][0]["summary"]["failed_idea_ids"] == ["metric_tradeoff"]
    assert memory["entries"][0]["summary"]["dataset_regressions"]["mmlu-redux"] == 2.5
    assert "dtype_patch" not in json.dumps(memory, ensure_ascii=False)
    assert (tmp_path / "method_memory.md").exists()


def test_shared_method_memory_default_global_skips_non_workspace_project(tmp_path: Path) -> None:
    config = {"orchestration": {"shared_method_memory": {"enabled": True}}}
    project_root = tmp_path / "proj"
    (project_root / "meta").mkdir(parents=True)
    (project_root / "meta" / "registry.yaml").write_text("research_topic: c2c\n", encoding="utf-8")

    result = append_shared_c2c_method_failure(
        config,
        project_root=project_root,
        performance_feedback={
            "summary": {"failure_class": "method_failure", "route": "proxy_rejected_same_direction", "iteration": 1},
            "candidate_results": [
                {
                    "id": "tmp_project_method",
                    "decision": "proxy_rejected",
                    "proxy_screen": {"status": "rejected", "proxy_dataset_deltas": {"mmlu-redux": -1.0}},
                }
            ],
        },
    )

    assert result == {"status": "skipped", "reason": "non_workspace_project_not_written_to_global_method_memory"}


def test_collect_used_shared_memory_refs_only_accepts_known_ids() -> None:
    memory = {
        "recent_entries": [
            {"memory_id": "mem_known_1"},
            {"memory_id": "mem_known_2"},
        ]
    }
    payload = {
        "used_shared_memory_refs": ["mem_known_1", "hallucinated"],
        "variant_candidates": [{"anti_repeat": "Avoid repeating failure from mem_known_2"}],
    }

    assert collect_used_shared_memory_refs(payload, memory) == ["mem_known_1"]
    assert collect_used_shared_memory_refs({"summary": "mem_known_2 influenced this"}, memory) == ["mem_known_2"]


def test_shared_method_memory_prioritizes_proxy_full_false_positives(tmp_path: Path) -> None:
    config = {
        "orchestration": {
            "shared_method_memory": {
                "enabled": True,
                "path": str(tmp_path / "method_memory.jsonl"),
                "summary_path": str(tmp_path / "method_memory.md"),
                "prompt_limit": 2,
            }
        }
    }
    low_project = tmp_path / "low"
    high_project = tmp_path / "high"
    for project in [low_project, high_project]:
        (project / "meta").mkdir(parents=True)
        (project / "meta" / "registry.yaml").write_text("research_topic: c2c\n", encoding="utf-8")
    append_shared_c2c_method_failure(
        config,
        project_root=low_project,
        performance_feedback={
            "summary": {"failure_class": "method_failure", "route": "proxy_rejected_same_direction", "iteration": 1},
            "candidate_results": [
                {
                    "id": "ordinary_proxy_fail",
                    "decision": "proxy_rejected",
                    "proxy_screen": {"status": "rejected", "proxy_dataset_deltas": {"mmlu-redux": -0.5}},
                }
            ],
        },
    )
    (high_project / "experiment" / "results").mkdir(parents=True)
    (high_project / "experiment" / "results" / "proxy_calibration.json").write_text(
        json.dumps(
            {
                "summary": {
                    "candidate_count": 1,
                    "proxy_false_positive_count": 1,
                    "proxy_false_positive_rate": 1.0,
                    "dataset_error_summary": {"mmlu-redux": {"misprediction_count": 1, "count": 1}},
                    "mechanism_false_positive_summary": {
                        "utility_predicted_cache_routing": {"count": 1, "false_positive_count": 1, "false_positive_rate": 1.0}
                    },
                    "method_feedback": {
                        "risky_datasets": [{"dataset": "mmlu-redux", "misprediction_count": 1}],
                        "risky_mechanisms": [{"mechanism_type": "utility_predicted_cache_routing", "false_positive_rate": 1.0}],
                        "recommendations": ["Downweight proxy false-positive mechanisms."],
                    },
                },
                "current_iteration": {
                    "iteration": 2,
                    "acceptance_passed": False,
                    "candidate_count": 1,
                    "proxy_false_positive_count": 1,
                    "proxy_false_positive_rate": 1.0,
                    "dataset_error_summary": {"mmlu-redux": {"misprediction_count": 1, "count": 1}},
                    "candidates": [
                        {
                            "id": "proxy_pass_full_fail",
                            "mechanism_type": "utility_predicted_cache_routing",
                            "proxy_false_positive": True,
                            "false_positive_reason": "proxy_mean_positive_full_mean_nonpositive",
                            "proxy_mean_delta": 0.8,
                            "full_mean_delta": -0.2,
                            "mispredicted_datasets": ["mmlu-redux"],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    appended = append_shared_c2c_method_failure(
        config,
        project_root=high_project,
        performance_feedback={
            "summary": {"failure_class": "method_failure", "route": "full_s3_failure", "iteration": 2},
            "candidate_results": [
                {
                    "id": "proxy_pass_full_fail",
                    "decision": "not_viable",
                    "metrics": {"mean": 49.8, "datasets": {"mmlu-redux": 48.5}},
                    "dataset_regressions": {"mmlu-redux": 1.5},
                    "proxy_screen": {"status": "passed", "proxy_delta_vs_baseline": 0.8},
                }
            ],
        },
    )

    memory = load_shared_method_memory(config)

    assert appended["status"] == "appended"
    assert appended["memory_priority"] > 5
    assert memory["entries"][0]["project_id"] == "high"
    quality = memory["entries"][0]["memory_quality"]
    assert set(quality["signals"]) >= {
        "dataset_regression",
        "full_train_failure",
        "proxy_dataset_misprediction",
        "proxy_full_false_positive",
        "proxy_risky_mechanism",
    }
    assert quality["evidence"]["failure_scope"] == "full_train"
    assert quality["score_components"]["full_train_failure"] > quality["score_components"].get("cheap_proxy_failure", 0)
    prompt_memory = shared_method_memory_for_prompt(config, limit=2)
    assert prompt_memory["ranking_policy"]["sort"] == "descending memory_quality.priority"
    assert prompt_memory["high_quality_memory_ids"][0] == memory["entries"][0]["memory_id"]
    assert prompt_memory["quality_summary"]["proxy_full_false_positive_memory_ids"] == [memory["entries"][0]["memory_id"]]
    assert prompt_memory["quality_summary"]["top_mispredicted_datasets"][0]["id"] == "mmlu-redux"
    assert prompt_memory["prompt_view"] == "catalog_only"
    assert prompt_memory["memory_catalog"][0]["memory_id"] == memory["entries"][0]["memory_id"]
    assert "cheap proxy looked positive but full train failed" in prompt_memory["memory_catalog"][0]["one_line_summary"]
    assert prompt_memory["memory_catalog"][0]["read_hint"]["query"].endswith(memory["entries"][0]["memory_id"])
    assert "proxy_calibration" not in prompt_memory["memory_catalog"][0]
    compact = memory["entries"][0]["proxy_calibration"]
    assert compact["false_positive_candidates"][0]["id"] == "proxy_pass_full_fail"
    assert "proxy_calibration" in json.dumps(memory["entries"], ensure_ascii=False)
    summary_md = (tmp_path / "method_memory.md").read_text(encoding="utf-8")
    assert "Proxy/full calibration" in summary_md


def test_shared_method_memory_quality_scores_repeated_ablation_and_cross_project_failures(tmp_path: Path) -> None:
    memory_path = tmp_path / "method_memory.jsonl"
    config = {
        "orchestration": {
            "shared_method_memory": {
                "enabled": True,
                "path": str(memory_path),
                "summary_path": str(tmp_path / "method_memory.md"),
                "prompt_limit": 10,
            }
        }
    }
    entries = [
        {
            "schema_version": "shared_method_failure_memory_v1",
            "timestamp": "2026-06-10T00:00:00Z",
            "memory_id": "shared_a",
            "project_id": "project_a",
            "route": "proxy_rejected_same_direction",
            "summary": {"dataset_regressions": {"ai2-arc": 0.7}},
            "entries": [
                {
                    "id": "candidate_a",
                    "decision": "proxy_rejected",
                    "mechanism_type": "shared_mechanism",
                    "proxy_screen": {"status": "rejected", "proxy_delta_vs_baseline": -0.4},
                }
            ],
            "direction_scorecard": {
                "mechanism_type": "shared_mechanism",
                "summary": {"same_direction_failure_count": 2, "attempt_count": 2},
                "attempts": [{"id": "a1"}, {"id": "a2"}],
            },
        },
        {
            "schema_version": "shared_method_failure_memory_v1",
            "timestamp": "2026-06-10T01:00:00Z",
            "memory_id": "shared_b",
            "project_id": "project_b",
            "route": "proxy_rejected_same_direction",
            "summary": {"dataset_regressions": {"mmlu-redux": 1.1}},
            "entries": [
                {
                    "id": "candidate_b",
                    "decision": "proxy_rejected",
                    "mechanism_type": "shared_mechanism",
                    "proxy_screen": {"status": "rejected", "proxy_delta_vs_baseline": -0.2},
                    "ablation_evidence": {"status": "completed", "enabled_disabled_delta": 0.0},
                }
            ],
            "direction_scorecard": {
                "mechanism_type": "shared_mechanism",
                "summary": {"same_direction_failure_count": 4, "attempt_count": 4},
                "attempts": [{"id": "b1"}, {"id": "b2"}, {"id": "b3"}, {"id": "b4"}],
            },
        },
        {
            "schema_version": "shared_method_failure_memory_v1",
            "timestamp": "2026-06-10T02:00:00Z",
            "memory_id": "single_c",
            "project_id": "project_c",
            "route": "proxy_rejected_same_direction",
            "summary": {},
            "entries": [
                {
                    "id": "candidate_c",
                    "decision": "proxy_rejected",
                    "mechanism_type": "single_project_mechanism",
                    "proxy_screen": {"status": "rejected", "proxy_delta_vs_baseline": -0.1},
                }
            ],
            "direction_scorecard": {"mechanism_type": "single_project_mechanism", "summary": {"attempt_count": 1}},
        },
    ]
    memory_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in entries) + "\n", encoding="utf-8")

    memory = load_shared_method_memory(config, limit=10)
    by_id = {item["memory_id"]: item for item in memory["entries"]}

    shared_quality = by_id["shared_b"]["memory_quality"]
    single_quality = by_id["single_c"]["memory_quality"]
    assert set(shared_quality["signals"]) >= {
        "ablation_evidence",
        "cheap_proxy_failure",
        "cross_project_mechanism_failure",
        "dataset_regression",
        "repeated_failure",
    }
    assert shared_quality["evidence"]["repeated_failure_count"] == 4
    assert shared_quality["evidence"]["cross_project_mechanisms"][0]["mechanism_type"] == "shared_mechanism"
    assert shared_quality["evidence"]["cross_project_mechanisms"][0]["project_count"] == 2
    assert shared_quality["priority"] > single_quality["priority"]
    assert "cross_project_mechanism_failure" not in single_quality["signals"]


def test_shared_method_memory_retrieves_relevant_top_k_by_context(tmp_path: Path) -> None:
    memory_path = tmp_path / "method_memory.jsonl"
    repo_fp = {
        "snapshot_path_hash": "repo_same",
        "allowed_files_hash": "files_same",
        "allowed_prefixes_hash": "prefix_same",
    }
    config = {
        "orchestration": {
            "shared_method_memory": {
                "enabled": True,
                "path": str(memory_path),
                "summary_path": str(tmp_path / "method_memory.md"),
                "prompt_limit": 2,
            }
        }
    }
    entries = [
        {
            "schema_version": "shared_method_failure_memory_v1",
            "timestamp": "2026-06-10T00:00:00Z",
            "memory_id": "generic_high_priority",
            "project_id": "old_high",
            "topic": "unrelated retrieval benchmark",
            "route": "full_s3_failure",
            "summary": {"dataset_regressions": {"unrelated-dataset": 3.0}},
            "entries": [
                {
                    "id": "unrelated",
                    "decision": "not_viable",
                    "mechanism_type": "unrelated_mechanism",
                    "metrics": {"mean": 1.0, "datasets": {"unrelated-dataset": 1.0}},
                    "dataset_regressions": {"unrelated-dataset": 3.0},
                    "ablation_evidence": {"status": "completed"},
                }
            ],
        },
        {
            "schema_version": "shared_method_failure_memory_v1",
            "timestamp": "2026-06-10T01:00:00Z",
            "memory_id": "relevant_dataset_mechanism",
            "project_id": "old_relevant",
            "topic": "cross tokenizer cache communication",
            "route": "proxy_rejected_same_direction",
            "summary": {"dataset_regressions": {"mmlu-redux": 1.2}},
            "entries": [
                {
                    "id": "relevant",
                    "decision": "proxy_rejected",
                    "mechanism_type": "utility_predicted_cache_routing",
                    "proxy_screen": {"status": "rejected", "proxy_dataset_deltas": {"mmlu-redux": -1.2}},
                }
            ],
            "source_repo_fingerprint": repo_fp,
        },
        {
            "schema_version": "shared_method_failure_memory_v1",
            "timestamp": "2026-06-10T02:00:00Z",
            "memory_id": "relevant_failure_mode",
            "project_id": "old_failure",
            "topic": "cross tokenizer cache communication",
            "route": "repairable_proxy_risk",
            "summary": {},
            "entries": [
                {
                    "id": "repairable",
                    "decision": "proxy_repairable",
                    "mechanism_type": "semantic_span_graph_alignment",
                    "proxy_screen": {"status": "repairable_proxy_risk", "proxy_delta_vs_baseline": -0.1},
                }
            ],
        },
    ]
    memory_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in entries) + "\n", encoding="utf-8")

    prompt_memory = shared_method_memory_for_prompt(
        config,
        limit=2,
        query_context={
            "topic": "cross tokenizer cache communication",
            "datasets": ["mmlu-redux"],
            "mechanism_types": ["utility_predicted_cache_routing"],
            "failure_modes": ["proxy_rejected"],
            "source_repo_fingerprint": repo_fp,
        },
    )

    assert prompt_memory["retrieval_policy"]["mode"] == "quality_weighted_top_k_retrieval"
    assert prompt_memory["retrieval_context"]["datasets"] == ["mmlu-redux"]
    assert [item["memory_id"] for item in prompt_memory["recent_entries"]] == [
        "relevant_dataset_mechanism",
        "relevant_failure_mode",
    ]
    first_retrieval = prompt_memory["recent_entries"][0]["retrieval"]
    assert set(first_retrieval["matched_fields"]) >= {"datasets", "mechanism_types", "source_repo_fingerprint"}
    assert first_retrieval["combined_score"] > prompt_memory["recent_entries"][0]["priority"]

import json
from pathlib import Path

from auto_research.failure_log import FailureLogManager, build_c2c_feedback_bundle, load_c2c_feedback_bundle


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
